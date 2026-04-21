#!/usr/bin/env python3
# coding=utf-8
"""Scale-only DDTree DPO finetuning for LittleBit-DFlash."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from specforge.core.dflash_dpo import (
    DFlashDPODataset,
    collate_dpo_records,
    compute_record_dpo_loss,
    count_record_pairs,
    resolve_dtype,
    set_littlebit_dpo_trainable,
)
from specforge.littlebit import (
    load_quantized_dflash_model,
    read_littlebit_config,
    save_quantized_dflash_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune a trained LittleBit-DFlash checkpoint with DDTree DPO."
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--target-model-path",
        type=str,
        default=None,
        help="Target model path for loading only embed_tokens/lm_head. Defaults to DPO manifest.",
    )
    model_group.add_argument("--draft-model-path", type=str, required=True)
    model_group.add_argument(
        "--ref-model-path",
        type=str,
        default=None,
        help="Frozen DPO reference. Defaults to --draft-model-path.",
    )
    model_group.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume policy weights and optimizer state from a previous DPO checkpoint.",
    )
    model_group.add_argument("--device", type=str, default="cuda")
    model_group.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    model_group.add_argument("--embedding-key", type=str, default=None)
    model_group.add_argument("--lm-head-key", type=str, default=None)
    model_group.add_argument("--trust-remote-code", action="store_true")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dpo-data-dir", type=str, required=True)
    data_group.add_argument("--dataloader-num-workers", type=int, default=2)
    data_group.add_argument(
        "--max-rounds-per-record",
        type=int,
        default=None,
        help="Optional cap for replayed rounds per sample.",
    )
    data_group.add_argument(
        "--max-pairs-per-round",
        type=int,
        default=None,
        help="Optional cap for DPO pairs consumed from each round.",
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=1)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=1e-5)
    training_group.add_argument("--warmup-ratio", type=float, default=0.03)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--weight-decay", type=float, default=0.0)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--max-num-steps", type=int, default=None)
    training_group.add_argument("--beta", type=float, default=0.2)
    training_group.add_argument("--label-smoothing", type=float, default=0.0)
    training_group.add_argument(
        "--detach-draft-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach replayed DFlash KV cache after each round to bound memory.",
    )
    training_group.add_argument(
        "--trainable-scope",
        type=str,
        default="littlebit_scales",
        choices=["littlebit_scales", "all_littlebit"],
        help="Default trains only LittleBit scale vectors and freezes sign factors.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--log-interval", type=int, default=10)
    output_group.add_argument("--save-interval", type=int, default=200)

    quant_group = parser.add_argument_group("littlebit fallback config")
    quant_group.add_argument("--quant-func", type=str, default="STEBinary")
    quant_group.add_argument("--split-dim", type=int, default=1024)
    quant_group.add_argument("--eff-bit", type=float, default=0.5)
    quant_group.add_argument("--kv-factor", type=float, default=1.0)
    quant_group.add_argument("--min-split-dim", type=int, default=8)
    quant_group.add_argument("--residual", action="store_true")

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    return parser.parse_args()


def distributed_requested() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed(args) -> tuple[bool, int, int, torch.device]:
    if distributed_requested():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=args.dist_timeout),
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, dist.get_rank(), dist.get_world_size(), device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    return False, 0, 1, device


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(message: str, *, rank: int) -> None:
    if rank == 0:
        print(message)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def sync_metric(value: float, *, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / dist.get_world_size()
    return float(tensor.item())


def sync_sum(value: float, *, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def apply_checkpoint_quant_config(args, checkpoint_path: str) -> None:
    quant_config = read_littlebit_config(checkpoint_path)
    for key, value in quant_config.items():
        setattr(args, key.replace("-", "_"), value)


def build_models(args, *, device: torch.device, torch_dtype: torch.dtype):
    policy_path = args.resume_from_checkpoint or args.draft_model_path
    ref_path = args.ref_model_path or args.draft_model_path

    apply_checkpoint_quant_config(args, policy_path)
    policy_model = load_quantized_dflash_model(
        policy_path,
        device=device,
        torch_dtype=torch_dtype,
        quant_args=args,
        do_train=False,
    )
    policy_model.config._attn_implementation = args.attention_backend
    policy_model.train()

    ref_model = load_quantized_dflash_model(
        ref_path,
        device=device,
        torch_dtype=torch_dtype,
        quant_args=args,
        do_train=False,
    )
    ref_model.config._attn_implementation = args.attention_backend
    ref_model.eval()
    ref_model.requires_grad_(False)

    trainable_info = set_littlebit_dpo_trainable(
        policy_model,
        trainable_scope=args.trainable_scope,
    )
    return policy_model, ref_model, trainable_info


def save_checkpoint(
    *,
    args,
    model,
    optimizer: BF16Optimizer,
    epoch: int,
    global_step: int,
    rank: int,
    distributed: bool,
) -> None:
    save_dir = Path(args.output_dir) / f"epoch_{epoch}_step_{global_step}"
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_quantized_dflash_model(unwrap_model(model), str(save_dir), args)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                **optimizer.state_dict(),
            },
            save_dir / "training_state.pt",
        )
        print(f"Saved checkpoint to {save_dir}")
    if distributed:
        dist.barrier()


def restore_optimizer_if_available(
    *,
    args,
    optimizer: BF16Optimizer,
    rank: int,
) -> tuple[int, int]:
    if not args.resume_from_checkpoint:
        return 0, 0
    state_path = Path(args.resume_from_checkpoint) / "training_state.pt"
    if not state_path.exists():
        print_rank0(
            f"No DPO training_state.pt found in {args.resume_from_checkpoint}; resuming weights only.",
            rank=rank,
        )
        return 0, 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state)
    start_epoch = int(state.get("epoch", 0))
    global_step = int(state.get("global_step", 0))
    print_rank0(
        f"Restored DPO optimizer state: epoch={start_epoch}, step={global_step}",
        rank=rank,
    )
    return start_epoch, global_step


def append_jsonl(path: Path, payload: Dict[str, Any], *, rank: int) -> None:
    if rank != 0:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    distributed, rank, world_size, device = setup_distributed(args)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = DFlashDPODataset(args.dpo_data_dir)
        manifest = dataset.manifest
        if args.target_model_path is None:
            args.target_model_path = manifest.get("target_model_path")
        if args.target_model_path is None:
            raise ValueError(
                "--target-model-path was not provided and the DPO manifest does not contain one."
            )

        torch_dtype = resolve_dtype(args.torch_dtype)
        policy_model, ref_model, trainable_info = build_models(
            args,
            device=device,
            torch_dtype=torch_dtype,
        )
        policy_unwrapped = policy_model
        block_size = int(manifest.get("block_size", getattr(policy_unwrapped, "block_size")))
        mask_token_id = int(
            manifest.get(
                "mask_token_id",
                getattr(policy_unwrapped, "mask_token_id", None),
            )
        )
        if block_size != int(getattr(policy_unwrapped, "block_size")):
            raise ValueError(
                f"DPO data block_size={block_size} does not match model block_size={policy_unwrapped.block_size}."
            )

        target_components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            embed_key=args.embedding_key,
            lm_head_key=args.lm_head_key,
            device=str(device),
            dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        target_components.eval()
        target_components.requires_grad_(False)

        print_rank0(
            "Loaded DPO data: "
            f"{len(dataset)} records, "
            f"{manifest.get('num_rounds')} rounds, "
            f"{manifest.get('num_pairs')} pairs",
            rank=rank,
        )
        print_rank0(
            "Trainable DPO parameters: "
            f"{trainable_info['num_trainable_parameters']} across "
            f"{trainable_info['num_trainable_tensors']} tensors "
            f"({args.trainable_scope})",
            rank=rank,
        )
        if trainable_info["num_trainable_parameters"] <= 0:
            raise ValueError(
                "No trainable LittleBit parameters were found. "
                "finetune_dpo.py expects a LittleBit-DFlash checkpoint."
            )

        if distributed:
            policy_model = DDP(
                policy_model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )

        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_dpo_records,
            pin_memory=False,
        )

        steps_per_epoch = max(math.ceil(len(dataloader) / args.accumulation_steps), 1)
        total_steps = steps_per_epoch * args.num_epochs
        if args.max_num_steps is not None:
            total_steps = min(total_steps, args.max_num_steps)
        optimizer = BF16Optimizer(
            policy_model,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=max(total_steps, 1),
        )
        start_epoch, global_step = restore_optimizer_if_available(
            args=args,
            optimizer=optimizer,
            rank=rank,
        )

        log_path = output_dir / "dpo_train_log.jsonl"
        append_jsonl(
            log_path,
            {
                "event": "start",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "args": vars(args),
                "manifest": manifest,
                "trainable_info": {
                    key: value
                    for key, value in trainable_info.items()
                    if key != "trainable_names"
                },
            },
            rank=rank,
        )

        micro_step = 0
        pending_backward = False
        last_log_time = time.time()
        stop_training = False
        latest_metrics: Dict[str, float] = {}
        epoch = start_epoch

        for epoch in range(start_epoch, args.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            iterator = tqdm(
                dataloader,
                desc=f"DDTree DPO Epoch {epoch}",
                leave=True,
                disable=rank != 0,
            )
            for records in iterator:
                if args.max_num_steps is not None and global_step >= args.max_num_steps:
                    stop_training = True
                    break

                batch_weighted_losses: List[torch.Tensor] = []
                batch_weight = 0.0
                local_metrics = {
                    "records": float(len(records)),
                    "records_with_pairs": 0.0,
                    "pairs": 0.0,
                    "weighted_pairs": 0.0,
                    "rounds": 0.0,
                    "rounds_with_pairs": 0.0,
                    "margin": 0.0,
                    "policy_logratio": 0.0,
                    "ref_logratio": 0.0,
                    "preference_accuracy": 0.0,
                }

                for record in records:
                    record_pair_count = count_record_pairs(record)
                    if record_pair_count <= 0:
                        continue
                    loss, metrics = compute_record_dpo_loss(
                        record=record,
                        policy_model=policy_model,
                        ref_model=ref_model,
                        embed_tokens=target_components.embed_tokens,
                        lm_head=target_components.lm_head,
                        block_size=block_size,
                        mask_token_id=mask_token_id,
                        device=device,
                        beta=args.beta,
                        label_smoothing=args.label_smoothing,
                        max_rounds_per_record=args.max_rounds_per_record,
                        max_pairs_per_round=args.max_pairs_per_round,
                        detach_draft_cache=args.detach_draft_cache,
                    )
                    local_metrics["rounds"] += metrics.get("rounds", 0.0)
                    local_metrics["rounds_with_pairs"] += metrics.get(
                        "rounds_with_pairs",
                        0.0,
                    )
                    local_metrics["pairs"] += metrics.get("pairs", 0.0)
                    local_metrics["weighted_pairs"] += metrics.get(
                        "weighted_pairs",
                        0.0,
                    )
                    if loss is None:
                        continue
                    local_metrics["records_with_pairs"] += 1.0
                    batch_weighted_losses.append(
                        loss * metrics.get("weighted_pairs", 0.0)
                    )
                    batch_weight += metrics.get("weighted_pairs", 0.0)
                    local_metrics["margin"] += metrics.get("margin", 0.0)
                    local_metrics["policy_logratio"] += metrics.get(
                        "policy_logratio",
                        0.0,
                    )
                    local_metrics["ref_logratio"] += metrics.get(
                        "ref_logratio",
                        0.0,
                    )
                    local_metrics["preference_accuracy"] += metrics.get(
                        "preference_accuracy",
                        0.0,
                    )

                if not batch_weighted_losses or batch_weight <= 0:
                    continue
                batch_loss = torch.stack(batch_weighted_losses).sum() / batch_weight
                (batch_loss / args.accumulation_steps).backward()
                pending_backward = True

                micro_step += 1
                if micro_step % args.accumulation_steps != 0:
                    continue

                optimizer.step()
                pending_backward = False
                global_step += 1

                denom = max(local_metrics["records_with_pairs"], 1.0)
                latest_metrics = {
                    "train/loss": sync_metric(
                        float(batch_loss.detach().item()),
                        device=device,
                        distributed=distributed,
                    ),
                    "train/lr": optimizer.get_learning_rate(),
                    "train/pairs": sync_sum(
                        local_metrics["pairs"],
                        device=device,
                        distributed=distributed,
                    ),
                    "train/weighted_pairs": sync_sum(
                        local_metrics["weighted_pairs"],
                        device=device,
                        distributed=distributed,
                    ),
                    "train/rounds": sync_sum(
                        local_metrics["rounds"],
                        device=device,
                        distributed=distributed,
                    ),
                    "train/records": sync_sum(
                        local_metrics["records"],
                        device=device,
                        distributed=distributed,
                    ),
                    "train/margin": sync_metric(
                        local_metrics["margin"] / denom,
                        device=device,
                        distributed=distributed,
                    ),
                    "train/policy_logratio": sync_metric(
                        local_metrics["policy_logratio"] / denom,
                        device=device,
                        distributed=distributed,
                    ),
                    "train/ref_logratio": sync_metric(
                        local_metrics["ref_logratio"] / denom,
                        device=device,
                        distributed=distributed,
                    ),
                    "train/preference_accuracy": sync_metric(
                        local_metrics["preference_accuracy"] / denom,
                        device=device,
                        distributed=distributed,
                    ),
                }

                if args.log_interval > 0 and global_step % args.log_interval == 0:
                    elapsed = time.time() - last_log_time
                    latest_metrics["train/step_time"] = elapsed
                    print_rank0(
                        "DPO - "
                        f"Step {global_step}, "
                        f"Loss: {latest_metrics['train/loss']:.4f}, "
                        f"Margin: {latest_metrics['train/margin']:.4f}, "
                        f"PrefAcc: {latest_metrics['train/preference_accuracy']:.4f}, "
                        f"Pairs: {latest_metrics['train/pairs']:.0f}, "
                        f"LR: {latest_metrics['train/lr']:.6g}",
                        rank=rank,
                    )
                    append_jsonl(
                        log_path,
                        {
                            "event": "train",
                            "step": global_step,
                            "epoch": epoch,
                            **latest_metrics,
                        },
                        rank=rank,
                    )
                    last_log_time = time.time()

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    save_checkpoint(
                        args=args,
                        model=policy_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        rank=rank,
                        distributed=distributed,
                    )

            if stop_training:
                break

        if pending_backward and (
            args.max_num_steps is None or global_step < args.max_num_steps
        ):
            optimizer.step()
            global_step += 1

        save_checkpoint(
            args=args,
            model=policy_model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            rank=rank,
            distributed=distributed,
        )
        append_jsonl(
            log_path,
            {
                "event": "finish",
                "step": global_step,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                **latest_metrics,
            },
            rank=rank,
        )
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
