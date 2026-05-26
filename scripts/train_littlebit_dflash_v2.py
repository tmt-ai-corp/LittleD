#!/usr/bin/env python3
# coding=utf-8
"""Train a LittleBit-converted DFlash model with the original DFlash CE loss.

This is a fallback QAT path:
pretrained DFlash -> LittleBit DualSVD init -> DFlash CE training.
It intentionally does not run QAKD, teacher DFlash, KL, or intermediate MSE.
"""

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
import torch.nn.functional as F
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
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


def optional_int(value):
    if value is None:
        return None
    if str(value).lower() == "none":
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LittleBit-DFlash with the DFlash CE fallback loss"
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument("--draft-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
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
    model_group.add_argument("--mask-token-id", type=int, default=None)
    model_group.add_argument("--num-anchors", type=int, default=512)
    model_group.add_argument(
        "--fixed-num-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep draft-token shape fixed at num_anchors * block_size.",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=7.0,
        help="DFlash token loss decay. Original DFlash examples use 7.0.",
    )
    model_group.add_argument("--embedding-key", type=str, default=None)
    model_group.add_argument("--lm-head-key", type=str, default=None)
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument(
        "--early-exit-hidden-mode",
        type=str,
        default="none",
        choices=["none", "copy", "add_embed", "gate"],
        help=(
            "How to impute missing prefill hidden chunks for early-exit prefill. "
            "'copy' repeats the last available chunk; 'add_embed' uses h + e_i; "
            "'gate' uses h * (1 + g_i)."
        ),
    )
    model_group.add_argument(
        "--early-exit-available-hidden-count",
        type=int,
        default=3,
        help=(
            "Number of target hidden chunks available during early-exit prefill. "
            "Chunks after this count are imputed only on prefill-mask positions."
        ),
    )
    model_group.add_argument(
        "--early-exit-prefill-mask-mode",
        type=str,
        default="loss_mask_zero",
        choices=["loss_mask_zero", "before_first_loss"],
        help=(
            "Which sequence positions should receive early-exit hidden imputation. "
            "'loss_mask_zero' covers prompt/non-loss tokens; 'before_first_loss' "
            "covers tokens before the first supervised answer token."
        ),
    )
    model_group.add_argument(
        "--draft-lm-head",
        action="store_true",
        help=(
            "Give the draft model its own LittleBit-quantized lm_head initialized "
            "from the target lm_head. When omitted, DFlash shares the target head."
        ),
    )
    model_group.add_argument(
        "--draft-lm-head-pruning",
        type=optional_int,
        default=None,
        help=(
            "If set with --draft-lm-head, prune the draft lm_head vocab to this "
            "size using train dataset loss-token statistics before LittleBit."
        ),
    )

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
    training_group.add_argument("--batch-size", type=int, default=2)
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
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument("--resume-from-checkpoint", type=str, default=None)
    training_group.add_argument(
        "--logit-chunk-size",
        type=int,
        default=512,
        help="Token chunk size for lm_head/CE. This preserves DFlash CE loss.",
    )
    training_group.add_argument("--debug-step-timing", action="store_true")
    training_group.add_argument("--debug-step-interval", type=int, default=1)

    quant_group = parser.add_argument_group("littlebit")
    quant_group.add_argument("--quant-mod", type=str, default="LittleBitLinear")
    quant_group.add_argument("--quant-func", type=str, default="STEBinary")
    quant_group.add_argument("--split-dim", type=int, default=1024)
    quant_group.add_argument("--eff-bit", type=float, default=0.5)
    quant_group.add_argument("--kv-factor", type=float, default=1.0)
    quant_group.add_argument("--min-split-dim", type=int, default=8)
    quant_group.add_argument("--group-size", type=int, default=128)
    quant_group.add_argument("--residual", action="store_true")
    quant_group.add_argument(
        "--lm-head-eff-bit",
        type=float,
        default=None,
        help="eff_bit for lm_head quantization. Defaults to --eff-bit if unset.",
    )
    quant_group.add_argument(
        "--lm-head-split-dim",
        type=int,
        default=None,
        help="split_dim for lm_head quantization. Defaults to --split-dim if unset.",
    )
    quant_group.add_argument(
        "--lm-head-residual",
        action="store_true",
        default=None,
        help="Use residual quantization for lm_head. Defaults to --residual if unset.",
    )
    quant_group.add_argument(
        "--lm-head-no-residual",
        action="store_false",
        dest="lm_head_residual",
    )
    quant_group.add_argument(
        "--lm-head-group-size",
        type=int,
        default=None,
        help="group_size for lm_head quantization. Defaults to --group-size if unset.",
    )
    quant_group.add_argument(
        "--lm-head-quant-func",
        type=str,
        default=None,
        help="quant_func for lm_head quantization. Defaults to --quant-func if unset.",
    )
    quant_group.add_argument(
        "--lm-head-quant-mod",
        type=str,
        default=None,
        help="quant_mod for lm_head quantization. Defaults to --quant-mod if unset.",
    )
    quant_group.add_argument(
        "--lm-head-min-split-dim",
        type=int,
        default=None,
        help="min_split_dim for lm_head quantization. Defaults to --min-split-dim.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=20)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def validate_draft_lm_head_args(args, target_vocab_size: int):
    if args.draft_lm_head_pruning is not None and not args.draft_lm_head:
        raise ValueError("--draft-lm-head-pruning requires --draft-lm-head.")
    if args.draft_lm_head_pruning is None:
        return
    if args.draft_lm_head_pruning <= 0:
        raise ValueError("--draft-lm-head-pruning must be positive.")
    if args.draft_lm_head_pruning > target_vocab_size:
        raise ValueError(
            "--draft-lm-head-pruning cannot exceed the target vocab size: "
            f"{args.draft_lm_head_pruning} > {target_vocab_size}."
        )


def get_draft_block_size(args, resume_checkpoint: Optional[str]) -> int:
    config_path = resume_checkpoint or args.draft_model_path
    config = AutoConfig.from_pretrained(
        config_path,
        trust_remote_code=args.trust_remote_code,
    )
    return int(getattr(config, "block_size"))


def build_dataloader(
    args,
    tokenizer,
    block_size: int,
    target_vocab_size: int,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[str]]:
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

    vocab_mapping_path = None
    if args.draft_lm_head_pruning is not None:
        vocab_cache_dir = os.path.join(args.cache_dir, "vocab_mapping")
        vocab_cache_key = f"{cache_key}-draft-lm-head-{args.draft_lm_head_pruning}"
        if not dist.is_initialized() or dist.get_rank() == 0:
            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=train_eagle3_dataset,
                target_vocab_size=target_vocab_size,
                draft_vocab_size=args.draft_lm_head_pruning,
                cache_dir=vocab_cache_dir,
                cache_key=vocab_cache_key,
            )
        if dist.is_initialized():
            dist.barrier()
        if vocab_mapping_path is None:
            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=train_eagle3_dataset,
                target_vocab_size=target_vocab_size,
                draft_vocab_size=args.draft_lm_head_pruning,
                cache_dir=vocab_cache_dir,
                cache_key=vocab_cache_key,
            )
        print_on_rank0(f"Draft lm_head vocab mapping: {vocab_mapping_path}")

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

    return train_dataloader, eval_dataloader, vocab_mapping_path


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
    return F.pad(tensor, (0, target_length - tensor.size(1)))


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


def build_lm_head_quant_kwargs(args):
    if not args.draft_lm_head:
        return None
    return {
        "eff_bit": args.lm_head_eff_bit
        if args.lm_head_eff_bit is not None
        else args.eff_bit,
        "split_dim": args.lm_head_split_dim
        if args.lm_head_split_dim is not None
        else args.split_dim,
        "residual": args.lm_head_residual
        if args.lm_head_residual is not None
        else args.residual,
        "group_size": args.lm_head_group_size
        if args.lm_head_group_size is not None
        else args.group_size,
        "quant_func": args.lm_head_quant_func
        if args.lm_head_quant_func is not None
        else args.quant_func,
        "quant_mod": args.lm_head_quant_mod
        if args.lm_head_quant_mod is not None
        else args.quant_mod,
        "min_split_dim": args.lm_head_min_split_dim
        if args.lm_head_min_split_dim is not None
        else args.min_split_dim,
    }


def load_draft_lm_head_vocab_mapping(vocab_mapping_path: Optional[str], device):
    if vocab_mapping_path is None:
        return None
    vocab_mapping = torch.load(vocab_mapping_path, map_location="cpu")
    t2d = vocab_mapping["t2d"].bool()
    draft_to_target = torch.nonzero(t2d, as_tuple=False).flatten().long()
    target_to_draft = torch.full((t2d.numel(),), -1, dtype=torch.long)
    target_to_draft[draft_to_target] = torch.arange(
        draft_to_target.numel(), dtype=torch.long
    )
    return {
        "t2d": t2d.to(device=device),
        "d2t": vocab_mapping["d2t"].long().to(device=device),
        "draft_to_target": draft_to_target.to(device=device),
        "target_to_draft": target_to_draft.to(device=device),
    }


def set_or_register_buffer(module, name: str, value: torch.Tensor):
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value)


def attach_draft_vocab_mapping(student_model, vocab_mapping):
    if vocab_mapping is None:
        return
    for name, value in vocab_mapping.items():
        set_or_register_buffer(student_model, name, value)
    student_model.vocab_mapping_loaded = True


def attach_draft_lm_head(student_model, target_components, args, vocab_mapping_path):
    """Attach a draft-owned lm_head before LittleBit conversion."""
    target_weight = target_components.lm_head.weight
    device = next(student_model.parameters()).device
    vocab_mapping = load_draft_lm_head_vocab_mapping(vocab_mapping_path, device)

    if vocab_mapping is None:
        head_weight = target_weight
        out_features = target_weight.shape[0]
    else:
        head_weight = target_weight.index_select(
            0, vocab_mapping["draft_to_target"].to(target_weight.device)
        )
        out_features = head_weight.shape[0]

    config = student_model.config
    lm_head = torch.nn.Linear(config.hidden_size, out_features, bias=False)
    if lm_head.weight.shape[1] != target_weight.shape[1]:
        raise ValueError(
            f"Draft lm_head hidden size {lm_head.weight.shape[1]} does not match "
            f"target lm_head hidden size {target_weight.shape[1]}."
        )
    lm_head = lm_head.to(device=device, dtype=target_weight.dtype)
    with torch.no_grad():
        lm_head.weight.copy_(head_weight.to(device=device))

    student_model.lm_head = lm_head
    student_model.draft_lm_head = True
    student_model.config.draft_lm_head = True
    student_model.config.draft_lm_head_pruning = args.draft_lm_head_pruning
    student_model.config.draft_lm_head_vocab_size = out_features
    args.draft_lm_head_vocab_size = out_features
    attach_draft_vocab_mapping(student_model, vocab_mapping)

    if args.draft_lm_head_pruning is None:
        print_on_rank0("Attached full draft lm_head initialized from target lm_head.")
    else:
        print_on_rank0(
            "Attached pruned draft lm_head initialized from target lm_head: "
            f"{target_weight.shape[0]} -> {out_features} vocab rows."
        )


def build_models(
    args,
    resume_checkpoint: Optional[str],
    target_components,
    vocab_mapping_path: Optional[str],
) -> Tuple[DFlashTargetModel, DFlashDraftModel]:
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

    if resume_checkpoint:
        resume_quant_config = read_littlebit_config(resume_checkpoint)
        for key, value in resume_quant_config.items():
            setattr(args, key.replace("-", "_"), value)
        validate_draft_lm_head_args(args, target_components.lm_head.out_features)
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
        if args.draft_lm_head:
            attach_draft_lm_head(
                student_model,
                target_components,
                args,
                vocab_mapping_path,
            )
        exclude_names = [] if args.draft_lm_head else ["lm_head"]
        student_model = apply_littlebit_patch(
            student_model,
            args,
            do_train=True,
            exclude_names=exclude_names,
            lm_head_kwargs=build_lm_head_quant_kwargs(args),
        )
        student_model = move_model_to_local_cuda(student_model)

    target_layer_ids = getattr(student_model, "target_layer_ids", None)
    if target_layer_ids is None:
        target_layer_ids = getattr(student_model.config, "target_layer_ids", None)
    if target_layer_ids is None:
        num_target_layers = getattr(student_model.config, "num_target_layers", None)
        num_draft_layers = getattr(student_model.config, "num_hidden_layers", None)
        if num_target_layers is not None and num_draft_layers is not None:
            from specforge.modeling.draft.dflash import build_target_layer_ids

            target_layer_ids = build_target_layer_ids(
                num_target_layers,
                num_draft_layers,
            )

    if target_layer_ids is not None and args.early_exit_hidden_mode != "none":
        if args.early_exit_available_hidden_count >= len(target_layer_ids):
            raise ValueError(
                "--early-exit-available-hidden-count must be smaller than the "
                "number of target_layer_ids when early-exit imputation is enabled."
            )
        if (
            not hasattr(student_model.config, "dflash_config")
            or student_model.config.dflash_config is None
        ):
            student_model.config.dflash_config = {}
        student_model.config.dflash_config[
            "early_exit_hidden_mode"
        ] = args.early_exit_hidden_mode
        student_model.config.dflash_config[
            "early_exit_available_hidden_count"
        ] = args.early_exit_available_hidden_count
        student_model.config.dflash_config[
            "early_exit_target_hidden_count"
        ] = len(target_layer_ids)
    elif hasattr(student_model.config, "dflash_config"):
        student_model.config.dflash_config["early_exit_hidden_mode"] = "none"

    student_model.config._attn_implementation = args.attention_backend
    student_model.train()
    target_model.set_capture_layers(student_model.target_layer_ids)
    return target_model, student_model


def save_checkpoint(args, epoch: int, step: int, student_model, optimizer):
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        save_quantized_dflash_model(unwrap_model(student_model), save_dir, args)
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


def build_dflash_helper(args, student_model, target_components, mask_token_id):
    student_unwrapped = unwrap_model(student_model)
    lm_head = (
        student_unwrapped.lm_head
        if getattr(student_unwrapped, "draft_lm_head", False)
        else target_components.lm_head
    )
    return OnlineDFlashModel(
        draft_model=student_unwrapped,
        target_lm_head=lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=student_unwrapped.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        fixed_num_anchors=args.fixed_num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        early_exit_hidden_mode=args.early_exit_hidden_mode,
        early_exit_available_hidden_count=args.early_exit_available_hidden_count,
        early_exit_prefill_mask_mode=args.early_exit_prefill_mask_mode,
    )


def compute_dflash_ce_from_batch(
    *,
    args,
    target_model,
    dflash_helper,
    student_model,
    input_ids,
    attention_mask,
    loss_mask,
    timer: Optional[StepTimer] = None,
):
    timer = timer or StepTimer(argparse.Namespace(debug_step_timing=False), 0)

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

    timer.start("student_draft_forward")
    student_output_hidden = dflash_helper.forward_hidden_from_view(
        hidden_states=hidden_states,
        training_view=training_view,
        draft_model=unwrap_model(student_model),
        output_hidden_states=False,
    )
    timer.end()

    timer.start("loss")
    loss, accuracy = dflash_helper.compute_loss_and_accuracy_from_hidden(
        student_output_hidden,
        training_view,
        chunk_size=args.logit_chunk_size,
    )
    timer.end()
    return loss, accuracy


def evaluate(
    eval_dataloader,
    *,
    target_model,
    dflash_helper,
    student_model,
    args,
    global_step,
    tracker,
):
    if eval_dataloader is None:
        return

    unwrap_model(student_model).eval()
    loss_sum = 0.0
    acc_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in eval_dataloader:
            input_ids, attention_mask, loss_mask = prepare_cuda_batch(data, args)
            loss, accuracy = compute_dflash_ce_from_batch(
                args=args,
                target_model=target_model,
                dflash_helper=dflash_helper,
                student_model=student_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            loss_sum += sync_scalar(loss)
            acc_sum += sync_scalar(accuracy)
            num_batches += 1

    if num_batches > 0:
        metrics = {
            "eval/loss": loss_sum / num_batches,
            "eval/accuracy": acc_sum / num_batches,
        }
        tracker.log(metrics, step=global_step)
        print_on_rank0(
            "Eval - "
            f"Step {global_step}, "
            f"Loss: {metrics['eval/loss']:.4f}, "
            f"Acc: {metrics['eval/accuracy']:.4f}"
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
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
        )
        target_components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            embed_key=args.embedding_key,
            lm_head_key=args.lm_head_key,
            device="cuda",
            trust_remote_code=args.trust_remote_code,
        )
        validate_draft_lm_head_args(args, target_components.lm_head.out_features)

        draft_block_size = get_draft_block_size(args, resume_checkpoint)
        train_dataloader, eval_dataloader, vocab_mapping_path = build_dataloader(
            args,
            tokenizer,
            draft_block_size,
            target_components.lm_head.out_features,
        )

        target_model, student_model = build_models(
            args,
            resume_checkpoint,
            target_components,
            vocab_mapping_path,
        )

        if dist.get_world_size() > 1:
            student_model = DDP(
                student_model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                find_unused_parameters=False,
            )

        student_unwrapped = unwrap_model(student_model)
        checkpoint_config = student_unwrapped.config
        mask_token_id = (
            args.mask_token_id
            if args.mask_token_id is not None
            else getattr(student_unwrapped, "mask_token_id", None)
        )
        if mask_token_id is None:
            dflash_config = getattr(checkpoint_config, "dflash_config", {}) or {}
            mask_token_id = dflash_config.get("mask_token_id", None)
        if mask_token_id is None and tokenizer.mask_token_id is not None:
            mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
            mask_token_id = tokenizer.mask_token_id

        student_unwrapped.mask_token_id = mask_token_id
        student_dflash_config = (
            getattr(student_unwrapped.config, "dflash_config", {}) or {}
        )
        student_dflash_config["mask_token_id"] = mask_token_id
        student_dflash_config["target_layer_ids"] = student_unwrapped.target_layer_ids
        student_unwrapped.config.dflash_config = student_dflash_config
        print_on_rank0(f"dflash_config: {student_dflash_config}")

        steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
        total_steps = args.num_epochs * steps_per_epoch
        if args.max_num_steps is not None:
            total_steps = min(total_steps, args.max_num_steps)
        print_on_rank0(f"Total training steps: {total_steps}")

        dflash_helper = build_dflash_helper(
            args, student_model, target_components, mask_token_id
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
                    train_dataloader, desc=f"LittleBit DFlash CE Epoch {epoch}", leave=True
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

                no_sync_context = (
                    student_model.no_sync()
                    if isinstance(student_model, DDP)
                    and global_step % args.accumulation_steps != 0
                    else nullcontext()
                )
                with no_sync_context:
                    loss, accuracy = compute_dflash_ce_from_batch(
                        args=args,
                        target_model=target_model,
                        dflash_helper=dflash_helper,
                        student_model=student_model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        loss_mask=loss_mask,
                        timer=timer,
                    )
                    timer.start("backward")
                    (loss / args.accumulation_steps).backward()
                    timer.end()

                if global_step % args.accumulation_steps == 0:
                    timer.start("optimizer_step")
                    optimizer.step()
                    timer.end()

                if global_step % args.log_interval == 0:
                    current_time = time.time()
                    logdict = {
                        "train/lr": optimizer.get_learning_rate(),
                        "train/loss": sync_scalar(loss),
                        "train/accuracy": sync_scalar(accuracy),
                        "train/step_time": current_time - last_time,
                    }
                    tracker.log(logdict, step=global_step)
                    print_on_rank0(
                        "Train - "
                        f"Step {global_step}, "
                        f"Loss: {logdict['train/loss']:.4f}, "
                        f"Acc: {logdict['train/accuracy']:.4f}, "
                        f"LR: {logdict['train/lr']:.6f}"
                    )
                    last_time = current_time

                if dist.get_rank() == 0 and hasattr(progress_bar, "set_postfix"):
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{accuracy.item():.4f}",
                        }
                    )

                if (
                    eval_dataloader is not None
                    and args.eval_interval > 0
                    and global_step % args.eval_interval == 0
                ):
                    evaluate(
                        eval_dataloader,
                        target_model=target_model,
                        dflash_helper=dflash_helper,
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
