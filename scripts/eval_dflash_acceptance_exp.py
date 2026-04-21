#!/usr/bin/env python3
# coding=utf-8
"""Experimental LittleBit/DFlash DDTree eval with deep tree and tensor stats."""

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from benchmarks.benchmarker import BENCHMARKS
from specforge.littlebit import load_quantized_dflash_model
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    build_ddtree_tree,
    compact_dynamic_cache,
    compile_ddtree_tree,
    extract_context_feature,
    find_first_stop_sequence,
    follow_verified_tree,
    sample,
)

try:
    from eval_dflash_acceptance import (
        build_prompt,
        build_stop_token_sequences,
        get_stop_strings,
        parse_benchmark_specs,
        resolve_device,
        resolve_dtype,
        truncate_at_stop_strings,
    )
except ImportError:
    from scripts.eval_dflash_acceptance import (
        build_prompt,
        build_stop_token_sequences,
        get_stop_strings,
        parse_benchmark_specs,
        resolve_device,
        resolve_dtype,
        truncate_at_stop_strings,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimental DFlash/DDTree evaluation with deep diagnostics"
    )
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-path", type=str, required=True)
    parser.add_argument(
        "--draft-type",
        type=str,
        default="dflash",
        choices=["dflash", "littlebit_dflash"],
    )
    parser.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        default=["mtbench:20", "gsm8k:20", "math500:20"],
        help="Format: <benchmark>:<num-samples>",
    )
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--apply_ddtree",
        action="store_true",
        default=False,
        help="Use DDTree verification on top of DFlash logits.",
    )
    parser.add_argument(
        "--ddtree_size",
        type=int,
        default=32,
        help="DDTree non-root node budget. DFlash-b16 still has max depth 16.",
    )
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--quant-func", type=str, default="STEBinary")
    parser.add_argument("--split-dim", type=int, default=1024)
    parser.add_argument("--eff-bit", type=float, default=0.5)
    parser.add_argument("--kv-factor", type=float, default=1.0)
    parser.add_argument("--min-split-dim", type=int, default=8)
    parser.add_argument("--residual", action="store_true")

    parser.add_argument(
        "--exp-tag",
        type=str,
        default="dflash_ddtree_exp",
        help="Tag stored in the JSON output for this experimental eval.",
    )
    parser.add_argument(
        "--option",
        type=str,
        default="1,2,3",
        help=(
            "Comma-separated experimental options to enable: "
            "1=DDTree structure stats, 2=leaf-batch DFlash redraft, "
            "3=hidden/logit tensor stats. Use 'all' or 'none' as aliases."
        ),
    )
    parser.add_argument(
        "--exp-leaf-batch-size",
        type=int,
        default=0,
        help="Chunk size for the extra leaf-batch DFlash pass. 0 means all leaves at once.",
    )
    parser.add_argument(
        "--exp-stat-sample-size",
        type=int,
        default=1_000_000,
        help="Maximum tensor values used for expensive quantile summaries.",
    )
    parser.add_argument(
        "--exp-logit-top-k",
        type=int,
        default=5,
        help="Number of top logit entries to sample per logged position.",
    )
    parser.add_argument(
        "--exp-disable-leaf-redraft",
        action="store_true",
        help="Skip the extra DFlash pass over DDTree leaf paths.",
    )
    args = parser.parse_args()
    try:
        exp_options = parse_exp_options(args.option)
    except ValueError as exc:
        parser.error(str(exc))
    if args.exp_disable_leaf_redraft:
        exp_options.discard(2)
    args.exp_options = sorted(exp_options)
    return args


def parse_exp_options(option_text: str) -> set[int]:
    text = (option_text or "").strip().lower()
    if text in {"all", "*"}:
        return {1, 2, 3}
    if text in {"", "none", "off", "false", "0"}:
        return set()

    options = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            option = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid --option value: {item!r}") from exc
        if option not in {1, 2, 3}:
            raise ValueError("--option only supports 1, 2, and 3")
        options.add(option)
    return options


def option_enabled(args, option: int) -> bool:
    return option in set(getattr(args, "exp_options", []))


def safe_float(value: Any) -> Optional[float]:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def summarize_numbers(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}

    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    mean = sum(sorted_values) / count
    variance = sum((value - mean) ** 2 for value in sorted_values) / count

    def percentile(pct: float) -> float:
        if count == 1:
            return sorted_values[0]
        rank = (count - 1) * pct
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return sorted_values[low]
        weight = rank - low
        return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight

    return {
        "count": count,
        "min": safe_float(sorted_values[0]),
        "max": safe_float(sorted_values[-1]),
        "mean": safe_float(mean),
        "std": safe_float(math.sqrt(variance)),
        "p05": safe_float(percentile(0.05)),
        "p25": safe_float(percentile(0.25)),
        "p50": safe_float(percentile(0.50)),
        "p75": safe_float(percentile(0.75)),
        "p95": safe_float(percentile(0.95)),
    }


def histogram(values: List[int]) -> Dict[str, int]:
    return {str(key): int(count) for key, count in sorted(Counter(values).items())}


def tensor_stats(
    tensor: Optional[torch.Tensor],
    *,
    stat_sample_size: int,
    quantiles: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0),
) -> Dict[str, Any]:
    if tensor is None:
        return {"is_none": True}

    detached = tensor.detach()
    out: Dict[str, Any] = {
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        return out

    values = detached.float().reshape(-1)
    finite_mask = torch.isfinite(values)
    finite_count = int(finite_mask.sum().item())
    out.update(
        {
            "finite_count": finite_count,
            "finite_fraction": safe_float(finite_count / max(int(values.numel()), 1)),
            "nan_count": int(torch.isnan(values).sum().item()),
            "posinf_count": int(torch.isposinf(values).sum().item()),
            "neginf_count": int(torch.isneginf(values).sum().item()),
        }
    )
    if finite_count == 0:
        return out

    finite = values[finite_mask]
    out.update(
        {
            "min": safe_float(finite.min().item()),
            "max": safe_float(finite.max().item()),
            "mean": safe_float(finite.mean().item()),
            "std": safe_float(finite.std(unbiased=False).item()),
            "rms": safe_float(torch.sqrt(torch.mean(finite * finite)).item()),
            "l1_mean": safe_float(finite.abs().mean().item()),
            "abs_max": safe_float(finite.abs().max().item()),
            "zero_fraction": safe_float((finite == 0).float().mean().item()),
            "abs_lt_1e_6_fraction": safe_float((finite.abs() < 1e-6).float().mean().item()),
            "abs_lt_1e_4_fraction": safe_float((finite.abs() < 1e-4).float().mean().item()),
            "abs_lt_1e_2_fraction": safe_float((finite.abs() < 1e-2).float().mean().item()),
        }
    )

    sample_values = finite
    max_values = max(int(stat_sample_size), 1)
    if finite.numel() > max_values:
        stride = math.ceil(finite.numel() / max_values)
        sample_values = finite[::stride][:max_values]
        out["quantiles_sampled"] = True
        out["quantiles_sample_size"] = int(sample_values.numel())
    else:
        out["quantiles_sampled"] = False
        out["quantiles_sample_size"] = int(sample_values.numel())

    q_tensor = torch.tensor(quantiles, device=sample_values.device, dtype=torch.float32)
    q_values = torch.quantile(sample_values, q_tensor)
    out["quantiles"] = {
        f"p{int(q * 100):02d}": safe_float(value.item())
        for q, value in zip(quantiles, q_values)
    }
    return out


def sampled_top_tokens(
    top_ids: torch.Tensor,
    top_values: torch.Tensor,
    top_probs: torch.Tensor,
    *,
    max_batches: int = 2,
    max_positions: int = 8,
) -> List[Dict[str, Any]]:
    rows = []
    top_ids_cpu = top_ids.detach().cpu()
    top_values_cpu = top_values.detach().float().cpu()
    top_probs_cpu = top_probs.detach().float().cpu()
    batch_count = min(int(top_ids_cpu.shape[0]), max_batches)
    pos_count = min(int(top_ids_cpu.shape[1]), max_positions)
    for batch_idx in range(batch_count):
        for pos_idx in range(pos_count):
            rows.append(
                {
                    "batch": batch_idx,
                    "position": pos_idx,
                    "token_ids": [int(item) for item in top_ids_cpu[batch_idx, pos_idx].tolist()],
                    "logits": [
                        safe_float(item)
                        for item in top_values_cpu[batch_idx, pos_idx].tolist()
                    ],
                    "probs": [
                        safe_float(item)
                        for item in top_probs_cpu[batch_idx, pos_idx].tolist()
                    ],
                }
            )
    return rows


def logit_stats(
    logits: torch.Tensor,
    *,
    stat_sample_size: int,
    top_k: int,
) -> Dict[str, Any]:
    out = tensor_stats(logits, stat_sample_size=stat_sample_size)
    if logits.numel() == 0 or logits.shape[-1] == 0:
        return out

    logits_float = logits.detach().float()
    log_probs = torch.log_softmax(logits_float, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top_count = min(max(int(top_k), 1), int(logits.shape[-1]))
    top_values, top_ids = torch.topk(logits_float, k=top_count, dim=-1)
    top_log_probs = log_probs.gather(dim=-1, index=top_ids)
    top_probs = top_log_probs.exp()
    margin = top_values[..., 0] - top_values[..., 1] if top_count > 1 else top_values[..., 0]

    out["distribution"] = {
        "entropy": tensor_stats(entropy, stat_sample_size=stat_sample_size),
        "top1_prob": tensor_stats(top_probs[..., 0], stat_sample_size=stat_sample_size),
        "top1_logit": tensor_stats(top_values[..., 0], stat_sample_size=stat_sample_size),
        "top1_margin": tensor_stats(margin, stat_sample_size=stat_sample_size),
    }
    out["top_tokens_sample"] = sampled_top_tokens(top_ids, top_values, top_probs)
    return out


def hidden_state_stack_stats(
    hidden_states: Optional[Tuple[torch.Tensor, ...]],
    *,
    stat_sample_size: int,
    selected_layer_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    if hidden_states is None:
        return []

    selected = set(selected_layer_ids or [])
    stack_stats = []
    for index, hidden_state in enumerate(hidden_states):
        entry = {
            "index": index,
            "target_layer_id": index - 1 if index > 0 else None,
            "selected_for_dflash": (index - 1) in selected if index > 0 else False,
            "stats": tensor_stats(hidden_state, stat_sample_size=stat_sample_size),
        }
        stack_stats.append(entry)
    return stack_stats


def build_children(parents: List[int]) -> List[List[int]]:
    children = [[] for _ in parents]
    for index, parent in enumerate(parents[1:], start=1):
        children[parent].append(index)
    return children


def path_to_root(index: int, parents: List[int]) -> List[int]:
    path = []
    while index >= 0:
        path.append(index)
        index = parents[index]
    path.reverse()
    return path


def compute_leaf_paths(
    *,
    root_token: torch.Tensor,
    node_token_ids: torch.Tensor,
    parents: List[int],
) -> List[Dict[str, Any]]:
    children = build_children(parents)
    tokens = [int(root_token.item())] + [int(token) for token in node_token_ids.tolist()]
    leaf_indices = [index for index, child_list in enumerate(children) if not child_list]

    leaf_paths = []
    for leaf_index in leaf_indices:
        path = path_to_root(leaf_index, parents)
        leaf_paths.append(
            {
                "leaf_index": int(leaf_index),
                "depth": int(len(path) - 1),
                "path_node_indices": [int(item) for item in path],
                "path_token_ids": [int(tokens[item]) for item in path],
            }
        )
    return leaf_paths


def tree_probability_stats(
    draft_logits: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    parents: List[int],
    leaf_indices: List[int],
    accepted_indices: List[int],
) -> Dict[str, Any]:
    if node_token_ids.numel() == 0:
        return {
            "local_logprob": summarize_numbers([]),
            "cumulative_logprob": summarize_numbers([]),
            "node_rank": summarize_numbers([]),
        }

    logits_float = draft_logits.detach().float()
    log_probs = torch.log_softmax(logits_float, dim=-1)
    local_logprobs = [0.0]
    cumulative_logprobs = [0.0 for _ in parents]
    ranks = [1]

    for node_index in range(1, len(parents)):
        token_id = int(node_token_ids[node_index - 1].item())
        depth = int(node_depths[node_index - 1].item())
        local_logprob = float(log_probs[depth - 1, token_id].item())
        parent = parents[node_index]
        local_logprobs.append(local_logprob)
        cumulative_logprobs[node_index] = cumulative_logprobs[parent] + local_logprob
        row = logits_float[depth - 1]
        ranks.append(int((row > row[token_id]).sum().item()) + 1)

    non_root_local = local_logprobs[1:]
    non_root_cumulative = cumulative_logprobs[1:]
    non_root_ranks = ranks[1:]
    leaf_cumulative = [cumulative_logprobs[index] for index in leaf_indices]
    accepted_cumulative = [cumulative_logprobs[index] for index in accepted_indices]
    accepted_local = [local_logprobs[index] for index in accepted_indices]
    accepted_ranks = [ranks[index] for index in accepted_indices]

    return {
        "local_logprob": summarize_numbers(non_root_local),
        "local_prob": summarize_numbers([math.exp(value) for value in non_root_local]),
        "cumulative_logprob": summarize_numbers(non_root_cumulative),
        "cumulative_prob": summarize_numbers([math.exp(value) for value in non_root_cumulative]),
        "leaf_cumulative_logprob": summarize_numbers(leaf_cumulative),
        "leaf_cumulative_prob": summarize_numbers([math.exp(value) for value in leaf_cumulative]),
        "node_rank": summarize_numbers([float(value) for value in non_root_ranks]),
        "node_rank_histogram": histogram(non_root_ranks),
        "accepted_cumulative_logprob": [safe_float(value) for value in accepted_cumulative],
        "accepted_cumulative_prob": [safe_float(math.exp(value)) for value in accepted_cumulative],
        "accepted_local_logprob": [safe_float(value) for value in accepted_local],
        "accepted_node_rank": [int(value) for value in accepted_ranks],
    }


def compute_tree_stats(
    *,
    round_index: int,
    start: int,
    block_size: int,
    tree_budget: int,
    root_token: torch.Tensor,
    verify_input_ids: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    parents: List[int],
    child_maps: List[Dict[int, int]],
    visibility_cpu: torch.Tensor,
    draft_logits: torch.Tensor,
    posterior: torch.Tensor,
    accepted_indices: List[int],
    next_token: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    del child_maps
    children = build_children(parents)
    node_count = len(parents)
    depths = [0] + [int(depth) for depth in node_depths.tolist()]
    tokens = [int(root_token.item())] + [int(token) for token in node_token_ids.tolist()]
    leaf_indices = [index for index, child_list in enumerate(children) if not child_list]
    leaf_depths = [depths[index] for index in leaf_indices]
    internal_indices = [index for index, child_list in enumerate(children) if child_list]
    branch_counts = [len(child_list) for child_list in children]
    internal_branch_counts = [len(children[index]) for index in internal_indices]
    accepted_depths = [depths[index] for index in accepted_indices]
    accepted_tokens = [tokens[index] for index in accepted_indices]
    accepted_last = accepted_indices[-1]
    accepted_ended_at_leaf = len(children[accepted_last]) == 0

    width_by_depth = Counter(depths)
    non_root_depths = depths[1:]
    visibility_rows = visibility_cpu.sum(dim=1).tolist()
    leaf_paths = []
    for leaf_index in leaf_indices:
        path = path_to_root(leaf_index, parents)
        leaf_paths.append(
            {
                "leaf_index": int(leaf_index),
                "depth": int(depths[leaf_index]),
                "path_node_indices": [int(item) for item in path],
                "path_token_ids": [int(tokens[item]) for item in path],
            }
        )

    posterior_tokens = posterior[0].detach().cpu().tolist()
    tree_stats = {
        "round_index": int(round_index),
        "decode_start_index": int(start),
        "block_size": int(block_size),
        "draft_horizon": int(block_size - 1),
        "tree_budget": int(tree_budget),
        "root_token_id": int(root_token.item()),
        "node_count_with_root": int(node_count),
        "non_root_node_count": int(max(node_count - 1, 0)),
        "max_depth": int(max(depths) if depths else 0),
        "non_root_depth": summarize_numbers([float(item) for item in non_root_depths]),
        "depth_histogram": histogram(non_root_depths),
        "width_by_depth_with_root": {str(key): int(width_by_depth[key]) for key in sorted(width_by_depth)},
        "max_width": int(max(width_by_depth.values()) if width_by_depth else 0),
        "mean_width": safe_float(sum(width_by_depth.values()) / max(len(width_by_depth), 1)),
        "leaf_count": int(len(leaf_indices)),
        "leaf_fraction": safe_float(len(leaf_indices) / max(node_count, 1)),
        "leaf_depth": summarize_numbers([float(item) for item in leaf_depths]),
        "leaf_depth_histogram": histogram(leaf_depths),
        "internal_node_count": int(len(internal_indices)),
        "branching_factor_all_nodes": summarize_numbers([float(item) for item in branch_counts]),
        "branching_factor_internal_nodes": summarize_numbers([float(item) for item in internal_branch_counts]),
        "branching_factor_histogram": histogram(branch_counts),
        "accepted_indices": [int(item) for item in accepted_indices],
        "accepted_token_ids": [int(item) for item in accepted_tokens],
        "accepted_path_length_with_root": int(len(accepted_indices)),
        "accepted_draft_token_count": int(max(len(accepted_indices) - 1, 0)),
        "accepted_path_depth": int(max(accepted_depths) if accepted_depths else 0),
        "accepted_path_depths": [int(item) for item in accepted_depths],
        "accepted_ended_at_leaf": bool(accepted_ended_at_leaf),
        "next_token_id": int(next_token),
        "posterior_token_ids_sample": [int(item) for item in posterior_tokens[: min(len(posterior_tokens), 16)]],
        "visibility_true_count": int(visibility_cpu.sum().item()),
        "visibility_density": safe_float(visibility_cpu.float().mean().item()),
        "visibility_row_true_count": summarize_numbers([float(item) for item in visibility_rows]),
        "leaf_paths": leaf_paths,
        "probability": tree_probability_stats(
            draft_logits,
            node_token_ids,
            node_depths,
            parents,
            leaf_indices,
            accepted_indices,
        ),
    }
    return tree_stats, leaf_paths


def run_leaf_redraft(
    *,
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    tree_target_hidden: torch.Tensor,
    verify_input_ids: torch.Tensor,
    leaf_paths: List[Dict[str, Any]],
    start: int,
    block_size: int,
    stat_sample_size: int,
    top_k: int,
    leaf_batch_size: int,
    collect_tensor_stats: bool,
) -> Dict[str, Any]:
    if not leaf_paths:
        return {"enabled": True, "leaf_count": 0, "groups": []}

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for path_info in leaf_paths:
        context_length = max(len(path_info["path_node_indices"]) - 1, 0)
        groups[context_length].append(path_info)

    redraft_stats: Dict[str, Any] = {
        "enabled": True,
        "leaf_count": int(len(leaf_paths)),
        "path_length_histogram": histogram(
            [len(path_info["path_node_indices"]) for path_info in leaf_paths]
        ),
        "groups": [],
    }
    target_device = next(target.parameters()).device

    for context_length, group_paths in sorted(groups.items()):
        chunk_size = len(group_paths) if leaf_batch_size <= 0 else max(leaf_batch_size, 1)
        for chunk_start in range(0, len(group_paths), chunk_size):
            chunk = group_paths[chunk_start : chunk_start + chunk_size]
            batch_size = len(chunk)
            leaf_input_ids = torch.full(
                (batch_size, block_size),
                model.mask_token_id,
                dtype=torch.long,
                device=target_device,
            )
            leaf_target_hidden = torch.empty(
                (batch_size, context_length, tree_target_hidden.shape[-1]),
                dtype=tree_target_hidden.dtype,
                device=tree_target_hidden.device,
            )
            for batch_index, path_info in enumerate(chunk):
                path_indices = path_info["path_node_indices"]
                context_indices = path_indices[:-1]
                leaf_index = path_indices[-1]
                leaf_input_ids[batch_index, 0] = verify_input_ids[0, leaf_index]
                if context_indices:
                    index_tensor = torch.tensor(
                        context_indices,
                        dtype=torch.long,
                        device=tree_target_hidden.device,
                    )
                    leaf_target_hidden[batch_index] = tree_target_hidden[0].index_select(
                        0,
                        index_tensor,
                    )

            position_ids = torch.arange(
                start,
                start + context_length + block_size,
                device=target_device,
            ).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
            noise_embedding = target.model.embed_tokens(leaf_input_ids)
            draft_output = model(
                target_hidden=leaf_target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=collect_tensor_stats,
                is_causal=False,
            )
            if collect_tensor_stats:
                draft_hidden, draft_hidden_states = draft_output
            else:
                draft_hidden = draft_output
                draft_hidden_states = None
            draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
            sampled_tokens = sample(draft_logits, temperature=0.0)
            group_stats = {
                "context_length": int(context_length),
                "path_length": int(context_length + 1),
                "chunk_start": int(chunk_start),
                "batch_size": int(batch_size),
                "leaf_indices": [int(item["leaf_index"]) for item in chunk],
                "leaf_input_ids": [
                    [int(token) for token in row]
                    for row in leaf_input_ids.detach().cpu().tolist()
                ],
                "sampled_token_ids": [
                    [int(token) for token in row]
                    for row in sampled_tokens.detach().cpu().tolist()
                ],
            }
            if collect_tensor_stats:
                group_stats.update(
                    {
                        "noise_embedding": tensor_stats(
                            noise_embedding,
                            stat_sample_size=stat_sample_size,
                        ),
                        "draft_hidden": tensor_stats(
                            draft_hidden,
                            stat_sample_size=stat_sample_size,
                        ),
                        "draft_hidden_states": hidden_state_stack_stats(
                            draft_hidden_states,
                            stat_sample_size=stat_sample_size,
                        ),
                        "draft_logits": logit_stats(
                            draft_logits,
                            stat_sample_size=stat_sample_size,
                            top_k=top_k,
                        ),
                    }
                )
            redraft_stats["groups"].append(group_stats)

    return redraft_stats


@torch.inference_mode()
def spec_generate_ddtree_exp(
    *,
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]],
    temperature: float,
    return_stats: bool,
    stop_token_sequences: Optional[List[List[int]]],
    tree_budget: int,
    stat_sample_size: int,
    top_k: int,
    leaf_batch_size: int,
    collect_tree_stats: bool,
    collect_tensor_stats: bool,
    run_leaf_redraft_enabled: bool,
):
    model.eval()
    block_size = model.block_size
    if block_size <= 1:
        return model.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            return_stats=return_stats,
            stop_token_sequences=stop_token_sequences,
            apply_ddtree=False,
        )

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = max(int(tree_budget), 0)
    max_tree_nodes = 1 + tree_budget
    target_dtype = getattr(target, "dtype", next(target.parameters()).dtype)

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        model.mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

    verify_input_ids_buffer = torch.empty(
        (1, max_tree_nodes),
        dtype=torch.long,
        device=target.device,
    )
    verify_position_ids_buffer = torch.empty(
        (1, max_tree_nodes),
        dtype=torch.long,
        device=target.device,
    )
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target_dtype,
        device=target.device,
    )
    tree_visibility_buffer = torch.empty(
        (max_tree_nodes, max_tree_nodes),
        dtype=torch.bool,
        device=target.device,
    )

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = time.perf_counter()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    time_to_first_token = time.perf_counter() - prefill_start

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits,
        temperature,
    )
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    acceptance_lengths = []
    stop_sequence_start = None
    previous_tree_start = 0
    previous_tree_length = 0
    ddtree_stage_times = {
        "draft": 0.0,
        "tree_build": 0.0,
        "tree_compile": 0.0,
        "verify": 0.0,
        "commit": 0.0,
        "leaf_redraft": 0.0,
    }
    exp_rounds = []
    prefill_stats = {"enabled": bool(collect_tensor_stats)}
    if collect_tensor_stats:
        prefill_stats.update(
            {
                "target_logits": logit_stats(
                    output.logits,
                    stat_sample_size=stat_sample_size,
                    top_k=top_k,
                ),
                "target_hidden_states": hidden_state_stack_stats(
                    output.hidden_states,
                    stat_sample_size=stat_sample_size,
                    selected_layer_ids=model.target_layer_ids,
                ),
                "dflash_context_feature": tensor_stats(
                    target_hidden,
                    stat_sample_size=stat_sample_size,
                ),
            }
        )

    start = input_ids.shape[1]
    round_index = 0
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        round_stats: Dict[str, Any] = {
            "round_index": int(round_index),
            "decode_start_index": int(start),
            "root_token_id": int(root_token[0, 0].item()),
        }

        stage_start = time.perf_counter()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_output = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            output_hidden_states=collect_tensor_stats,
            is_causal=False,
        )
        if collect_tensor_stats:
            draft_hidden, draft_hidden_states = draft_output
        else:
            draft_hidden = draft_output
            draft_hidden_states = None
        draft_logits = target.lm_head(draft_hidden[:, -draft_horizon:, :])
        past_key_values_draft.crop(start)
        ddtree_stage_times["draft"] += time.perf_counter() - stage_start
        round_stats["draft"] = {
            "block_output_ids_before_tree": [
                int(token) for token in block_output_ids[0].detach().cpu().tolist()
            ],
        }
        if collect_tensor_stats:
            round_stats["draft"].update(
                {
                    "noise_embedding": tensor_stats(
                        noise_embedding,
                        stat_sample_size=stat_sample_size,
                    ),
                    "draft_hidden": tensor_stats(
                        draft_hidden,
                        stat_sample_size=stat_sample_size,
                    ),
                    "draft_hidden_states": hidden_state_stack_stats(
                        draft_hidden_states,
                        stat_sample_size=stat_sample_size,
                    ),
                    "draft_logits": logit_stats(
                        draft_logits,
                        stat_sample_size=stat_sample_size,
                        top_k=top_k,
                    ),
                }
            )

        stage_start = time.perf_counter()
        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility_cpu,
        ) = build_ddtree_tree(draft_logits[0], tree_budget)
        ddtree_stage_times["tree_build"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        (
            verify_input_ids,
            verify_position_ids,
            verify_attention_mask,
            previous_tree_start,
            previous_tree_length,
        ) = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target_dtype,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        ddtree_stage_times["tree_compile"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        ddtree_stage_times["verify"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(
            accepted_indices,
            dtype=torch.long,
            device=verify_input_ids.device,
        )
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        tree_target_hidden = extract_context_feature(
            output.hidden_states,
            model.target_layer_ids,
        )
        target_hidden = tree_target_hidden.index_select(1, accepted_index_tensor)

        if collect_tree_stats:
            tree_stats, leaf_paths = compute_tree_stats(
                round_index=round_index,
                start=start,
                block_size=block_size,
                tree_budget=tree_budget,
                root_token=root_token[0, 0],
                verify_input_ids=verify_input_ids,
                node_token_ids=node_token_ids,
                node_depths=node_depths,
                parents=parents,
                child_maps=child_maps,
                visibility_cpu=visibility_cpu,
                draft_logits=draft_logits[0],
                posterior=posterior,
                accepted_indices=accepted_indices,
                next_token=next_token,
            )
            round_stats["tree"] = tree_stats
        else:
            leaf_paths = (
                compute_leaf_paths(
                    root_token=root_token[0, 0],
                    node_token_ids=node_token_ids,
                    parents=parents,
                )
                if run_leaf_redraft_enabled
                else []
            )
            round_stats["tree"] = {
                "enabled": False,
                "node_count_with_root": int(len(parents)),
            }
        round_stats["verify"] = {
            "verify_input_ids": [
                int(token) for token in verify_input_ids[0].detach().cpu().tolist()
            ],
            "verify_position_ids": [
                int(pos) for pos in verify_position_ids[0].detach().cpu().tolist()
            ],
            "posterior_token_ids": [
                int(token) for token in posterior[0].detach().cpu().tolist()
            ],
        }
        if collect_tensor_stats:
            round_stats["verify"].update(
                {
                    "attention_mask": tensor_stats(
                        verify_attention_mask,
                        stat_sample_size=stat_sample_size,
                    ),
                    "target_logits": logit_stats(
                        output.logits,
                        stat_sample_size=stat_sample_size,
                        top_k=top_k,
                    ),
                    "target_hidden_states": hidden_state_stack_stats(
                        output.hidden_states,
                        stat_sample_size=stat_sample_size,
                        selected_layer_ids=model.target_layer_ids,
                    ),
                    "tree_target_hidden": tensor_stats(
                        tree_target_hidden,
                        stat_sample_size=stat_sample_size,
                    ),
                }
            )

        if run_leaf_redraft_enabled:
            leaf_stage_start = time.perf_counter()
            round_stats["leaf_redraft"] = run_leaf_redraft(
                model=model,
                target=target,
                tree_target_hidden=tree_target_hidden,
                verify_input_ids=verify_input_ids,
                leaf_paths=leaf_paths,
                start=start,
                block_size=block_size,
                stat_sample_size=stat_sample_size,
                top_k=top_k,
                leaf_batch_size=leaf_batch_size,
                collect_tensor_stats=collect_tensor_stats,
            )
            ddtree_stage_times["leaf_redraft"] += time.perf_counter() - leaf_stage_start
        else:
            round_stats["leaf_redraft"] = {"enabled": False}

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        ddtree_stage_times["commit"] += time.perf_counter() - stage_start
        round_stats["commit"] = {
            "accepted_length_with_root": int(len(accepted_indices)),
            "accepted_tokens": [
                int(token) for token in accepted_tokens[0].detach().cpu().tolist()
            ],
            "next_token_id": int(next_token),
            "new_decode_start_index": int(start),
        }
        if collect_tensor_stats:
            round_stats["commit"]["next_context_feature"] = tensor_stats(
                target_hidden,
                stat_sample_size=stat_sample_size,
            )
        exp_rounds.append(round_stats)

        search_end = min(start + 1, max_length, output_ids.shape[1])
        stop_sequence_index = find_first_stop_sequence(
            output_ids[0, num_input_tokens:search_end],
            stop_token_sequences,
        )
        if stop_sequence_index is not None:
            stop_sequence_start = num_input_tokens + stop_sequence_index
            break

        if stop_token_ids is not None:
            stop_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            if torch.isin(output_ids[0, num_input_tokens:search_end], stop_ids).any():
                break

        round_index += 1

    output_ids = output_ids[:, :max_length]
    if stop_sequence_start is not None:
        output_ids = output_ids[:, :stop_sequence_start]
    output_ids = output_ids[:, output_ids[0] != model.mask_token_id]
    if stop_sequence_start is None and stop_token_ids is not None:
        stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:],
            stop_token_ids_tensor,
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_new_tokens = max(output_ids.shape[1] - num_input_tokens, 0)
    accept_length = (
        sum(acceptance_lengths) / len(acceptance_lengths)
        if acceptance_lengths
        else 1.0
    )
    return output_ids, {
        "acceptance_lengths": acceptance_lengths,
        "num_new_tokens": num_new_tokens,
        "num_speculation_steps": len(acceptance_lengths),
        "accept_length": accept_length,
        "time_to_first_token": time_to_first_token,
        "exp_option_enabled": {
            "1_tree_stats": bool(collect_tree_stats),
            "2_leaf_redraft": bool(run_leaf_redraft_enabled),
            "3_hidden_logit_stats": bool(collect_tensor_stats),
        },
        "ddtree_size": tree_budget,
        "ddtree_stage_times": ddtree_stage_times,
        "exp_prefill": prefill_stats,
        "exp_rounds": exp_rounds,
    }


def aggregate_exp_stats(turn_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    tree_depths = []
    tree_widths = []
    leaf_counts = []
    leaf_depths = []
    accepted_lengths = []
    accepted_depths = []
    accepted_leaf_flags = []
    node_counts = []

    for stats in turn_stats:
        for round_stats in stats.get("exp_rounds", []):
            tree = round_stats.get("tree", {})
            if tree.get("enabled") is False:
                continue
            tree_depths.append(float(tree.get("max_depth", 0)))
            tree_widths.append(float(tree.get("max_width", 0)))
            leaf_counts.append(float(tree.get("leaf_count", 0)))
            node_counts.append(float(tree.get("node_count_with_root", 0)))
            accepted_lengths.append(float(tree.get("accepted_path_length_with_root", 0)))
            accepted_depths.append(float(tree.get("accepted_path_depth", 0)))
            accepted_leaf_flags.append(1.0 if tree.get("accepted_ended_at_leaf") else 0.0)
            leaf_depth = tree.get("leaf_depth", {})
            if leaf_depth.get("count", 0):
                leaf_depths.append(float(leaf_depth.get("mean", 0.0)))

    return {
        "round_count": int(len(accepted_lengths)),
        "tree_max_depth": summarize_numbers(tree_depths),
        "tree_max_width": summarize_numbers(tree_widths),
        "tree_node_count_with_root": summarize_numbers(node_counts),
        "leaf_count": summarize_numbers(leaf_counts),
        "leaf_depth_mean_per_round": summarize_numbers(leaf_depths),
        "accepted_path_length_with_root": summarize_numbers(accepted_lengths),
        "accepted_path_depth": summarize_numbers(accepted_depths),
        "accepted_ended_at_leaf_fraction": safe_float(
            sum(accepted_leaf_flags) / len(accepted_leaf_flags)
        )
        if accepted_leaf_flags
        else None,
    }


def run_generation(
    *,
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    args,
    stop_strings: Optional[List[str]] = None,
):
    target_device = next(target_model.parameters()).device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = encoded["input_ids"].to(target_device)
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    stop_strings = stop_strings or []
    stop_token_sequences = build_stop_token_sequences(tokenizer, stop_strings)

    if args.apply_ddtree:
        output_ids, stats = spec_generate_ddtree_exp(
            model=draft_model,
            target=target_model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            return_stats=True,
            stop_token_sequences=stop_token_sequences,
            tree_budget=args.ddtree_size,
            stat_sample_size=args.exp_stat_sample_size,
            top_k=args.exp_logit_top_k,
            leaf_batch_size=args.exp_leaf_batch_size,
            collect_tree_stats=option_enabled(args, 1),
            collect_tensor_stats=option_enabled(args, 3),
            run_leaf_redraft_enabled=option_enabled(args, 2),
        )
    else:
        output_ids, stats = draft_model.spec_generate(
            target=target_model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            return_stats=True,
            stop_token_sequences=stop_token_sequences,
            apply_ddtree=False,
        )

    generated = tokenizer.decode(
        output_ids[0][input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    generated = truncate_at_stop_strings(generated, stop_strings)
    return generated, stats


def evaluate_benchmark(
    *,
    benchmark_name: str,
    benchmarker,
    draft_model,
    target_model,
    tokenizer,
    args,
):
    questions, labels = benchmarker.load_data()
    predictions = []
    total_latency = 0.0
    total_new_tokens = 0
    total_spec_steps = 0
    prompt_results = []
    all_turn_stats = []

    for question, label in zip(questions, labels):
        turn_outputs = []
        turn_stats = []
        stop_strings = get_stop_strings(benchmark_name)
        start = time.perf_counter()

        if benchmark_name == "mtbench":
            for _turn_idx in range(2):
                prompt = build_prompt(
                    benchmark_name,
                    benchmarker,
                    question,
                    tokenizer,
                    generated_answers=turn_outputs,
                )
                generated, stats = run_generation(
                    draft_model=draft_model,
                    target_model=target_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=args.max_new_tokens
                    or benchmarker.get_max_new_tokens(),
                    temperature=args.temperature,
                    stop_strings=stop_strings,
                    args=args,
                )
                turn_outputs.append(generated)
                turn_stats.append(stats)
        else:
            prompt = build_prompt(benchmark_name, benchmarker, question, tokenizer)
            generated, stats = run_generation(
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens or benchmarker.get_max_new_tokens(),
                temperature=args.temperature,
                stop_strings=stop_strings,
                args=args,
            )
            turn_outputs.append(generated)
            turn_stats.append(stats)

        elapsed = time.perf_counter() - start
        total_latency += elapsed

        prompt_new_tokens = sum(item["num_new_tokens"] for item in turn_stats)
        prompt_steps = sum(item["num_speculation_steps"] for item in turn_stats)
        total_new_tokens += prompt_new_tokens
        total_spec_steps += prompt_steps
        all_turn_stats.extend(turn_stats)

        if benchmark_name == "mtbench":
            predictions.append(turn_outputs)
        else:
            predictions.append(benchmarker.extract_answer(turn_outputs[0], label))

        prompt_results.append(
            {
                "question": question,
                "outputs": turn_outputs,
                "latency": elapsed,
                "num_new_tokens": prompt_new_tokens,
                "num_speculation_steps": prompt_steps,
                "accept_length": prompt_new_tokens / prompt_steps if prompt_steps else 1.0,
                "turn_stats": turn_stats,
                "exp_summary": aggregate_exp_stats(turn_stats),
            }
        )

    accuracy = None
    if labels and any(label is not None for label in labels) and benchmark_name != "mtbench":
        accuracy = benchmarker.compute_accuracy(predictions, labels)

    return {
        "benchmark": benchmark_name,
        "num_samples": len(questions),
        "latency": total_latency,
        "num_new_tokens": total_new_tokens,
        "num_speculation_steps": total_spec_steps,
        "accept_length": total_new_tokens / total_spec_steps if total_spec_steps else 1.0,
        "output_throughput": total_new_tokens / total_latency if total_latency > 0 else 0.0,
        "accuracy": accuracy,
        "generation_mode": "ddtree_exp" if args.apply_ddtree else "dflash",
        "ddtree_size": args.ddtree_size if args.apply_ddtree else None,
        "exp_tag": args.exp_tag,
        "exp_options": args.exp_options,
        "exp_summary": aggregate_exp_stats(all_turn_stats),
        "prompts": prompt_results,
        "note": (
            "Experimental diagnostics include per-round DDTree structure, "
            "leaf-batch DFlash redraft stats, and hidden/logit summaries."
        ),
    }


def main():
    args = parse_args()
    torch_dtype = resolve_dtype(args.torch_dtype)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    target_load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.apply_ddtree:
        target_load_kwargs["attn_implementation"] = "sdpa"
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        **target_load_kwargs,
    ).to(device).eval()

    if args.draft_type == "littlebit_dflash":
        draft_model = load_quantized_dflash_model(
            args.draft_model_path,
            device=device,
            torch_dtype=torch_dtype,
            quant_args=args,
        )
    else:
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        draft_model.eval()

    results = {
        "exp_tag": args.exp_tag,
        "target_model_path": args.target_model_path,
        "draft_model_path": args.draft_model_path,
        "draft_type": args.draft_type,
        "apply_ddtree": args.apply_ddtree,
        "ddtree_size": args.ddtree_size if args.apply_ddtree else None,
        "exp_options": args.exp_options,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "benchmarks": [],
    }

    for benchmark_name, num_samples in parse_benchmark_specs(args.benchmark_list):
        benchmarker_cls = BENCHMARKS.get(benchmark_name)
        benchmarker = benchmarker_cls(num_samples=num_samples)
        benchmark_result = evaluate_benchmark(
            benchmark_name=benchmark_name,
            benchmarker=benchmarker,
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            args=args,
        )
        results["benchmarks"].append(benchmark_result)
        print(
            f"{benchmark_name}: "
            f"tag={args.exp_tag}, "
            f"option={','.join(str(item) for item in args.exp_options) or 'none'}, "
            f"mode={'ddtree_exp' if args.apply_ddtree else 'dflash'}, "
            f"accept_length={benchmark_result['accept_length']:.3f}, "
            f"throughput={benchmark_result['output_throughput']:.3f}, "
            f"accuracy={benchmark_result['accuracy']}"
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved experimental results to {output_path}")


if __name__ == "__main__":
    main()
