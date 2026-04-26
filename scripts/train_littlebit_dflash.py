#!/usr/bin/env python3
# coding=utf-8
"""Train a LittleBit-quantized DFlash model with QAKD."""

import argparse
import hashlib
import logging
import math
import os
import time
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.core.littlebit_dflash import (
    compute_littlebit_dflash_losses_from_hidden,
)
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.littlebit import (
    apply_littlebit_patch,
    load_quantized_dflash_model,
    read_littlebit_config,
    save_quantized_dflash_model,
)
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LittleBit-DFlash model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument("--draft-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target hidden-state generation.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend used by the DFlash draft model.",
    )
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Override MASK token id. Defaults to checkpoint/tokenizer value.",
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions to sample per sequence.",
    )
    model_group.add_argument(
        "--fixed-num-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DFlash draft-token shape fixed at num_anchors * block_size. "
        "This avoids FlexAttention recompilation and DDP stragglers on variable-length batches.",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Optional decay for DFlash token weights. This affects logged CE only.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Embedding weight key in the target model.",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="LM head weight key in the target model.",
    )
    model_group.add_argument("--trust-remote-code", action="store_true")

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=3)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=2e-5)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument(
        "--pad-to-max-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pad every batch to max_length to reduce per-rank shape drift.",
    )
    training_group.add_argument("--warmup-ratio", type=float, default=0.03)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--max-num-steps", type=int, default=None)
    training_group.add_argument(
        "--l2l-loss-scale",
        type=float,
        default=10.0,
        help="Intermediate hidden-state MSE scale, following LittleBit.",
    )
    training_group.add_argument(
        "--kd-loss-scale",
        type=float,
        default=1.0,
        help="Final-logit KL scale.",
    )
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument("--resume-from-checkpoint", type=str, default=None)
    training_group.add_argument(
        "--logit-chunk-size",
        type=int,
        default=512,
        help="Token chunk size for lm_head/KL/CE calculation. "
        "Smaller values reduce peak memory for large num_anchors.",
    )
    training_group.add_argument(
        "--debug-step-timing",
        action="store_true",
        help="Print per-rank timings around each expensive training phase.",
    )
    training_group.add_argument(
        "--debug-step-interval",
        type=int,
        default=1,
        help="Print debug timings every N global steps when debug-step-timing is enabled.",
    )

    quant_group = parser.add_argument_group("littlebit")
    quant_group.add_argument("--quant-mod", type=str, default="LittleBitLinear")
    quant_group.add_argument("--quant-func", type=str, default="STEBinary")
    quant_group.add_argument("--split-dim", type=int, default=1024)
    quant_group.add_argument("--eff-bit", type=float, default=0.5)
    quant_group.add_argument("--kv-factor", type=float, default=1.0)
    quant_group.add_argument("--min-split-dim", type=int, default=8)
    quant_group.add_argument("--group-size", type=int, default=128)
    quant_group.add_argument("--residual", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism used by the target backend.",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_dataloader(
    args, tokenizer, block_size: int
) -> Tuple[DataLoader, Optional[DataLoader]]:
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    min_loss_tokens = 2 * block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def get_resume_checkpoint(args) -> Optional[str]:
    if args.resume_from_checkpoint:
        return args.resume_from_checkpoint
    if args.resume and os.path.isdir(args.output_dir):
        last_checkpoint, _ = get_last_checkpoint(args.output_dir)
        return last_checkpoint
    return None


def sync_scalar(value: torch.Tensor) -> float:
    value = value.detach().float()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value.item()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def move_model_to_local_cuda(model):
    local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return model.to(device=local_device, dtype=torch.bfloat16)


def pad_2d_to_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    if tensor.size(1) == target_length:
        return tensor
    if tensor.size(1) > target_length:
        return tensor[:, :target_length].contiguous()
    return torch.nn.functional.pad(tensor, (0, target_length - tensor.size(1)))


def prepare_cuda_batch(data, args):
    input_ids = data["input_ids"].cuda(non_blocking=True)
    attention_mask = data["attention_mask"].cuda(non_blocking=True)
    loss_mask = data["loss_mask"].cuda(non_blocking=True)

    if args.pad_to_max_length:
        input_ids = pad_2d_to_length(input_ids, args.max_length)
        attention_mask = pad_2d_to_length(attention_mask, args.max_length)
        loss_mask = pad_2d_to_length(loss_mask, args.max_length)

    return input_ids, attention_mask, loss_mask


class StepTimer:
    def __init__(self, args, global_step: int):
        self.enabled = (
            args.debug_step_timing
            and args.debug_step_interval > 0
            and global_step % args.debug_step_interval == 0
        )
        self.global_step = global_step
        self.phase = None
        self.start_time = None

    def start(self, phase: str):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.phase = phase
        self.start_time = time.time()
        print_with_rank(f"Step {self.global_step}: begin {phase}")

    def end(self):
        if not self.enabled or self.phase is None:
            return
        torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        print_with_rank(
            f"Step {self.global_step}: end {self.phase} ({elapsed:.3f}s)"
        )
        self.phase = None
        self.start_time = None


def build_models(
    args,
    resume_checkpoint: Optional[str],
) -> Tuple[DFlashTargetModel, DFlashDraftModel, torch.nn.Module]:
    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    teacher_model = DFlashDraftModel.from_pretrained(
        args.draft_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).cuda()
    teacher_model.config._attn_implementation = args.attention_backend
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    if resume_checkpoint:
        resume_quant_config = read_littlebit_config(resume_checkpoint)
        for key, value in resume_quant_config.items():
            setattr(args, key.replace("-", "_"), value)
        student_model = load_quantized_dflash_model(
            resume_checkpoint,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
            torch_dtype=torch.bfloat16,
            quant_args=args,
            do_train=True,
        )
    else:
        student_model = DFlashDraftModel.from_pretrained(
            args.draft_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        ).cuda()
        student_model = apply_littlebit_patch(student_model, args, do_train=True)
        student_model = move_model_to_local_cuda(student_model)

    student_model.config._attn_implementation = args.attention_backend
    student_model.train()
    return target_model, teacher_model, student_model


def save_checkpoint(
    args,
    epoch: int,
    step: int,
    student_model,
    optimizer: BF16Optimizer,
):
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        student_to_save = unwrap_model(student_model)
        save_quantized_dflash_model(student_to_save, save_dir, args)
        torch.save(
            {
                "epoch": epoch,
                "global_step": step,
                "args": vars(args),
                **optimizer.state_dict(),
            },
            os.path.join(save_dir, "training_state.pt"),
        )
        print_on_rank0(f"Saved checkpoint to {save_dir}")
    dist.barrier()


def evaluate(
    eval_dataloader,
    *,
    target_model,
    dflash_helper,
    teacher_model,
    student_model,
    args,
    global_step,
    tracker,
):
    if eval_dataloader is None:
        return

    unwrap_model(student_model).eval()
    metrics = {
        "eval/loss": 0.0,
        "eval/kd_loss": 0.0,
        "eval/l2l_loss": 0.0,
        "eval/task_loss": 0.0,
        "eval/task_accuracy": 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for data in eval_dataloader:
            input_ids, attention_mask, loss_mask = prepare_cuda_batch(data, args)
            target_output = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = target_output.hidden_states.cuda()
            training_view = dflash_helper.prepare_training_view(
                input_ids=input_ids,
                loss_mask=loss_mask,
            )

            teacher_output_hidden, teacher_hidden_states = (
                dflash_helper.forward_hidden_from_view(
                    hidden_states=hidden_states,
                    training_view=training_view,
                    draft_model=teacher_model,
                    output_hidden_states=True,
                )
            )
            student_output_hidden, student_hidden_states = (
                dflash_helper.forward_hidden_from_view(
                    hidden_states=hidden_states,
                    training_view=training_view,
                    draft_model=student_model,
                    output_hidden_states=True,
                )
            )
            losses = compute_littlebit_dflash_losses_from_hidden(
                student_output_hidden=student_output_hidden,
                teacher_output_hidden=teacher_output_hidden,
                student_hidden_states=student_hidden_states,
                teacher_hidden_states=teacher_hidden_states,
                lm_head=dflash_helper.lm_head,
                l2l_loss_scale=args.l2l_loss_scale,
                kd_loss_scale=args.kd_loss_scale,
                logit_chunk_size=args.logit_chunk_size,
            )
            task_loss, task_accuracy = dflash_helper.compute_loss_and_accuracy_from_hidden(
                student_output_hidden,
                training_view,
                chunk_size=args.logit_chunk_size,
            )

            metrics["eval/loss"] += sync_scalar(losses.loss)
            metrics["eval/kd_loss"] += sync_scalar(losses.kd_loss)
            metrics["eval/l2l_loss"] += sync_scalar(losses.l2l_loss)
            metrics["eval/task_loss"] += sync_scalar(task_loss)
            metrics["eval/task_accuracy"] += sync_scalar(task_accuracy)
            num_batches += 1

    if num_batches > 0:
        for key in metrics:
            metrics[key] /= num_batches
        tracker.log(metrics, step=global_step)
        print_on_rank0(
            "Eval - "
            f"Step {global_step}, "
            f"Loss: {metrics['eval/loss']:.4f}, "
            f"KD: {metrics['eval/kd_loss']:.4f}, "
            f"MSE: {metrics['eval/l2l_loss']:.4f}, "
            f"TaskLoss: {metrics['eval/task_loss']:.4f}, "
            f"TaskAcc: {metrics['eval/task_accuracy']:.4f}"
        )

    unwrap_model(student_model).train()


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    resume_checkpoint = get_resume_checkpoint(args)
    if resume_checkpoint:
        print_on_rank0(f"Resuming from checkpoint: {resume_checkpoint}")

    try:
        target_model, teacher_model, student_model = build_models(
            args, resume_checkpoint
        )
        target_model.set_capture_layers(teacher_model.target_layer_ids)

        if dist.get_world_size() > 1:
            student_model = DDP(
                student_model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                find_unused_parameters=False,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
        )

        checkpoint_config = unwrap_model(student_model).config
        mask_token_id = (
            args.mask_token_id
            if args.mask_token_id is not None
            else getattr(unwrap_model(student_model), "mask_token_id", None)
        )
        if mask_token_id is None:
            dflash_config = getattr(checkpoint_config, "dflash_config", {}) or {}
            mask_token_id = dflash_config.get("mask_token_id", None)
        if mask_token_id is None and tokenizer.mask_token_id is not None:
            mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
            mask_token_id = tokenizer.mask_token_id
        student_unwrapped = unwrap_model(student_model)
        student_unwrapped.mask_token_id = mask_token_id
        teacher_model.mask_token_id = mask_token_id
        student_dflash_config = (
            getattr(student_unwrapped.config, "dflash_config", {}) or {}
        )
        student_dflash_config["mask_token_id"] = mask_token_id
        student_dflash_config["target_layer_ids"] = teacher_model.target_layer_ids
        student_unwrapped.config.dflash_config = student_dflash_config

        train_dataloader, eval_dataloader = build_dataloader(
            args, tokenizer, student_unwrapped.block_size
        )
        steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
        total_steps = args.num_epochs * steps_per_epoch
        if args.max_num_steps is not None:
            total_steps = min(total_steps, args.max_num_steps)
        print_on_rank0(f"Total training steps: {total_steps}")

        target_components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            embed_key=args.embedding_key,
            lm_head_key=args.lm_head_key,
            device="cuda",
            trust_remote_code=args.trust_remote_code,
        )

        dflash_helper = OnlineDFlashModel(
            draft_model=student_unwrapped,
            target_lm_head=target_components.lm_head,
            target_embed_tokens=target_components.embed_tokens,
            block_size=student_unwrapped.block_size,
            mask_token_id=mask_token_id,
            attention_backend=args.attention_backend,
            num_anchors=args.num_anchors,
            fixed_num_anchors=args.fixed_num_anchors,
            loss_decay_gamma=args.loss_decay_gamma,
        )

        optimizer = BF16Optimizer(
            student_model,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=max(total_steps, 1),
        )

        start_epoch = 0
        global_step = 0
        if resume_checkpoint:
            training_state_path = os.path.join(resume_checkpoint, "training_state.pt")
            if os.path.exists(training_state_path):
                resume_state = torch.load(
                    training_state_path, map_location="cpu", weights_only=False
                )
                optimizer.load_state_dict(resume_state)
                start_epoch = resume_state.get("epoch", 0)
                global_step = resume_state.get("global_step", 0)
                print_on_rank0(
                    f"Restored optimizer state: epoch={start_epoch}, step={global_step}"
                )
        skip_steps = global_step - start_epoch * len(train_dataloader)

        tracker = create_tracker(args, args.output_dir)
        print_on_rank0("Tracker initialized successfully.")

        last_time = time.time()
        stop_training = False

        for epoch in range(start_epoch, args.num_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            unwrap_model(student_model).train()

            if dist.get_rank() == 0:
                progress_bar = tqdm(
                    train_dataloader, desc=f"QAT Epoch {epoch}", leave=True
                )
            else:
                progress_bar = train_dataloader

            for step_in_epoch, data in enumerate(progress_bar):
                if epoch == start_epoch and step_in_epoch < skip_steps:
                    continue
                if args.max_num_steps is not None and global_step >= args.max_num_steps:
                    stop_training = True
                    break

                global_step += 1
                timer = StepTimer(args, global_step)

                timer.start("batch_to_cuda")
                input_ids, attention_mask, loss_mask = prepare_cuda_batch(data, args)
                timer.end()

                timer.start("target_forward")
                target_output = target_model.generate_dflash_data(
                    input_ids, attention_mask, loss_mask
                )
                hidden_states = target_output.hidden_states.cuda()
                timer.end()

                timer.start("prepare_training_view")
                training_view = dflash_helper.prepare_training_view(
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                )
                timer.end()

                no_sync_context = (
                    student_model.no_sync()
                    if isinstance(student_model, DDP)
                    and global_step % args.accumulation_steps != 0
                    else nullcontext()
                )
                with no_sync_context:
                    timer.start("teacher_draft_forward")
                    with torch.no_grad():
                        teacher_output_hidden, teacher_hidden_states = (
                            dflash_helper.forward_hidden_from_view(
                                hidden_states=hidden_states,
                                training_view=training_view,
                                draft_model=teacher_model,
                                output_hidden_states=True,
                            )
                        )
                    timer.end()

                    timer.start("student_draft_forward")
                    student_output_hidden, student_hidden_states = (
                        dflash_helper.forward_hidden_from_view(
                            hidden_states=hidden_states,
                            training_view=training_view,
                            draft_model=student_model,
                            output_hidden_states=True,
                        )
                    )
                    timer.end()

                    timer.start("loss_and_backward")
                    losses = compute_littlebit_dflash_losses_from_hidden(
                        student_output_hidden=student_output_hidden,
                        teacher_output_hidden=teacher_output_hidden,
                        student_hidden_states=student_hidden_states,
                        teacher_hidden_states=teacher_hidden_states,
                        lm_head=dflash_helper.lm_head,
                        l2l_loss_scale=args.l2l_loss_scale,
                        kd_loss_scale=args.kd_loss_scale,
                        logit_chunk_size=args.logit_chunk_size,
                    )
                    task_loss, task_accuracy = (
                        dflash_helper.compute_loss_and_accuracy_from_hidden(
                            student_output_hidden,
                            training_view,
                            chunk_size=args.logit_chunk_size,
                        )
                    )
                    (losses.loss / args.accumulation_steps).backward()
                    timer.end()

                if global_step % args.accumulation_steps == 0:
                    timer.start("optimizer_step")
                    optimizer.step()
                    timer.end()

                if global_step % args.log_interval == 0:
                    current_time = time.time()
                    logdict = {
                        "train/lr": optimizer.get_learning_rate(),
                        "train/loss": sync_scalar(losses.loss),
                        "train/kd_loss": sync_scalar(losses.kd_loss),
                        "train/l2l_loss": sync_scalar(losses.l2l_loss),
                        "train/task_loss": sync_scalar(task_loss),
                        "train/task_accuracy": sync_scalar(task_accuracy),
                        "train/step_time": current_time - last_time,
                    }
                    tracker.log(logdict, step=global_step)
                    print_on_rank0(
                        "Train - "
                        f"Step {global_step}, "
                        f"Loss: {logdict['train/loss']:.4f}, "
                        f"KD: {logdict['train/kd_loss']:.4f}, "
                        f"MSE: {logdict['train/l2l_loss']:.4f}, "
                        f"TaskLoss: {logdict['train/task_loss']:.4f}, "
                        f"TaskAcc: {logdict['train/task_accuracy']:.4f}, "
                        f"LR: {logdict['train/lr']:.6f}"
                    )
                    last_time = current_time

                if (
                    eval_dataloader is not None
                    and args.eval_interval > 0
                    and global_step % args.eval_interval == 0
                ):
                    evaluate(
                        eval_dataloader,
                        target_model=target_model,
                        dflash_helper=dflash_helper,
                        teacher_model=teacher_model,
                        student_model=student_model,
                        args=args,
                        global_step=global_step,
                        tracker=tracker,
                    )

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    save_checkpoint(args, epoch, global_step, student_model, optimizer)

            if stop_training:
                break

        save_checkpoint(args, epoch, global_step, student_model, optimizer)
        tracker.close()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
