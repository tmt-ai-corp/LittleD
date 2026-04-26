# coding=utf-8
"""DDTree preference data and DPO helpers for DFlash draft models."""

from __future__ import annotations

import bisect
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import DynamicCache

from specforge.littlebit.modules import LittleBitLinear

DPO_DATA_VERSION = 1

PAIR_KIND_ACCEPTED_SIBLING = 0
PAIR_KIND_TERMINAL_FALLBACK = 1


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def compact_tree_summary(
    *,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    parents: List[int],
    accepted_indices: List[int],
    next_token: int,
    posterior: torch.Tensor,
    draft_logits: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Return a compact, tensor-friendly tree summary for offline analysis."""
    child_counts = Counter(parent for parent in parents[1:])
    leaf_indices = [
        index
        for index in range(len(parents))
        if index != 0 and child_counts.get(index, 0) == 0
    ]
    summary: Dict[str, Any] = {
        "node_token_ids": node_token_ids.detach().cpu().to(torch.long),
        "node_depths": node_depths.detach().cpu().to(torch.long),
        "parents": torch.tensor(parents, dtype=torch.long),
        "accepted_indices": torch.tensor(accepted_indices, dtype=torch.long),
        "accepted_length_with_root": int(len(accepted_indices)),
        "accepted_draft_token_count": max(int(len(accepted_indices)) - 1, 0),
        "next_token_id": int(next_token),
        "posterior_token_ids": posterior.detach().cpu().to(torch.long).view(-1),
        "leaf_indices": torch.tensor(leaf_indices, dtype=torch.long),
        "branching_histogram": dict(sorted(Counter(child_counts.values()).items())),
    }
    if draft_logits is not None and node_token_ids.numel() > 0:
        depths = node_depths.to(draft_logits.device) - 1
        tokens = node_token_ids.to(draft_logits.device)
        selected_logits = draft_logits.index_select(0, depths)
        node_log_probs = selected_logits.gather(1, tokens.view(-1, 1)).squeeze(1)
        node_log_probs = node_log_probs - torch.logsumexp(selected_logits, dim=-1)
        summary["node_log_probs"] = node_log_probs.detach().cpu().float()
    return summary


def build_position_preference_pairs(
    *,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    child_maps: List[Dict[int, int]],
    accepted_indices: List[int],
    next_token: int,
    block_size: int,
    include_terminal_pairs: bool = True,
    max_pairs_per_round: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Build factorized DDTree DPO pairs from accepted edges and siblings.

    Each pair compares the target-accepted token at a DFlash depth against a
    sibling branch token at the same depth. If enabled, the terminal target
    posterior token is also compared against children of the terminal node when
    the accepted path stops before a candidate child.
    """
    depths: List[int] = []
    chosen: List[int] = []
    rejected: List[int] = []
    chosen_node_indices: List[int] = []
    rejected_node_indices: List[int] = []
    parent_node_indices: List[int] = []
    pair_kinds: List[int] = []

    def append_pair(
        *,
        depth: int,
        chosen_token: int,
        rejected_token: int,
        chosen_node_index: int,
        rejected_node_index: int,
        parent_node_index: int,
        pair_kind: int,
    ) -> None:
        if depth <= 0 or depth >= block_size:
            return
        if chosen_token == rejected_token:
            return
        depths.append(int(depth))
        chosen.append(int(chosen_token))
        rejected.append(int(rejected_token))
        chosen_node_indices.append(int(chosen_node_index))
        rejected_node_indices.append(int(rejected_node_index))
        parent_node_indices.append(int(parent_node_index))
        pair_kinds.append(int(pair_kind))

    for path_offset in range(1, len(accepted_indices)):
        parent_index = int(accepted_indices[path_offset - 1])
        chosen_index = int(accepted_indices[path_offset])
        chosen_token = int(node_token_ids[chosen_index - 1])
        chosen_depth = int(node_depths[chosen_index - 1])
        for sibling_token, sibling_index in child_maps[parent_index].items():
            if int(sibling_index) == chosen_index:
                continue
            append_pair(
                depth=chosen_depth,
                chosen_token=chosen_token,
                rejected_token=int(sibling_token),
                chosen_node_index=chosen_index,
                rejected_node_index=int(sibling_index),
                parent_node_index=parent_index,
                pair_kind=PAIR_KIND_ACCEPTED_SIBLING,
            )

    if include_terminal_pairs and accepted_indices:
        terminal_parent = int(accepted_indices[-1])
        terminal_children = child_maps[terminal_parent]
        if terminal_parent == 0:
            terminal_depth = 1
        else:
            terminal_depth = int(node_depths[terminal_parent - 1]) + 1
        for rejected_token, rejected_index in terminal_children.items():
            append_pair(
                depth=terminal_depth,
                chosen_token=int(next_token),
                rejected_token=int(rejected_token),
                chosen_node_index=-1,
                rejected_node_index=int(rejected_index),
                parent_node_index=terminal_parent,
                pair_kind=PAIR_KIND_TERMINAL_FALLBACK,
            )

    if max_pairs_per_round is not None and max_pairs_per_round > 0:
        depths = depths[:max_pairs_per_round]
        chosen = chosen[:max_pairs_per_round]
        rejected = rejected[:max_pairs_per_round]
        chosen_node_indices = chosen_node_indices[:max_pairs_per_round]
        rejected_node_indices = rejected_node_indices[:max_pairs_per_round]
        parent_node_indices = parent_node_indices[:max_pairs_per_round]
        pair_kinds = pair_kinds[:max_pairs_per_round]

    pair_count = len(depths)
    return {
        "depths": torch.tensor(depths, dtype=torch.long),
        "chosen_token_ids": torch.tensor(chosen, dtype=torch.long),
        "rejected_token_ids": torch.tensor(rejected, dtype=torch.long),
        "chosen_node_indices": torch.tensor(chosen_node_indices, dtype=torch.long),
        "rejected_node_indices": torch.tensor(rejected_node_indices, dtype=torch.long),
        "parent_node_indices": torch.tensor(parent_node_indices, dtype=torch.long),
        "pair_kinds": torch.tensor(pair_kinds, dtype=torch.long),
        "weights": torch.ones(pair_count, dtype=torch.float32),
    }


def count_record_pairs(record: Dict[str, Any]) -> int:
    total = 0
    for round_item in record.get("rounds", []):
        pairs = round_item.get("pairs", {})
        depths = pairs.get("depths")
        if isinstance(depths, torch.Tensor):
            total += int(depths.numel())
        elif depths is not None:
            total += len(depths)
    return total


def count_record_rounds(record: Dict[str, Any]) -> int:
    return len(record.get("rounds", []))


def detach_dynamic_cache_(cache: DynamicCache) -> None:
    """Detach cached draft KV tensors to keep TreePO memory bounded."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache = [
            tensor.detach() if tensor is not None else None for tensor in cache.key_cache
        ]
        cache.value_cache = [
            tensor.detach() if tensor is not None else None
            for tensor in cache.value_cache
        ]
        return
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys.detach()
            if getattr(layer, "values", None) is not None:
                layer.values = layer.values.detach()


def set_littlebit_dpo_trainable(
    model: nn.Module,
    *,
    trainable_scope: str = "littlebit_scales",
) -> Dict[str, Any]:
    """Freeze a LittleBit-DFlash model and enable only the requested DPO params."""
    for param in model.parameters():
        param.requires_grad = False

    scale_names = {
        "u1",
        "u2",
        "v1",
        "v2",
        "u1_R",
        "u2_R",
        "v1_R",
        "v2_R",
        "U_scale",
        "V_scale",
        "U_R_scale",
        "V_R_scale",
    }
    trainable_names: List[str] = []
    total_trainable = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, LittleBitLinear):
            continue
        if trainable_scope == "littlebit_scales":
            allowed = scale_names
        elif trainable_scope == "all_littlebit":
            allowed = None
        else:
            raise ValueError(f"Unsupported trainable scope: {trainable_scope}")

        for param_name, param in module.named_parameters(recurse=False):
            if allowed is None or param_name in allowed:
                param.requires_grad = True
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                trainable_names.append(full_name)
                total_trainable += int(param.numel())

    return {
        "trainable_scope": trainable_scope,
        "num_trainable_tensors": len(trainable_names),
        "num_trainable_parameters": total_trainable,
        "trainable_names": trainable_names,
    }


class DFlashDPODataset(Dataset):
    """Lazy dataset over prepare_dpo_data.py shard files."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing DPO manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("version") != DPO_DATA_VERSION:
            raise ValueError(
                f"Unsupported DPO data version: {self.manifest.get('version')}"
            )

        self.shards = self.manifest.get("shards", [])
        if not self.shards:
            raise ValueError(f"No shards listed in {manifest_path}")
        self.cumulative: List[int] = []
        running = 0
        for shard in self.shards:
            running += int(shard["num_records"])
            self.cumulative.append(running)
        self._cached_shard_index: Optional[int] = None
        self._cached_records: Optional[List[Dict[str, Any]]] = None

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _load_shard(self, shard_index: int) -> List[Dict[str, Any]]:
        if self._cached_shard_index == shard_index and self._cached_records is not None:
            return self._cached_records
        shard_path = self.data_dir / self.shards[shard_index]["path"]
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        if payload.get("version") != DPO_DATA_VERSION:
            raise ValueError(f"Unsupported shard version in {shard_path}")
        records = payload["records"]
        self._cached_shard_index = shard_index
        self._cached_records = records
        return records

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index = len(self) + index
        shard_index = bisect.bisect_right(self.cumulative, index)
        shard_start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        records = self._load_shard(shard_index)
        return records[index - shard_start]


def collate_dpo_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return records


def _round_pair_tensors(
    round_item: Dict[str, Any],
    *,
    device: torch.device,
    max_pairs_per_round: Optional[int],
    loss_decay_gamma: Optional[float],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    pairs = round_item.get("pairs", {})
    depths = pairs.get("depths")
    if depths is None:
        return None
    if not isinstance(depths, torch.Tensor):
        depths = torch.tensor(depths, dtype=torch.long)
    if depths.numel() == 0:
        return None

    chosen = pairs["chosen_token_ids"]
    rejected = pairs["rejected_token_ids"]
    weights = pairs.get("weights")
    if not isinstance(chosen, torch.Tensor):
        chosen = torch.tensor(chosen, dtype=torch.long)
    if not isinstance(rejected, torch.Tensor):
        rejected = torch.tensor(rejected, dtype=torch.long)
    if weights is None:
        weights = torch.ones_like(depths, dtype=torch.float32)
    elif not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32)

    if max_pairs_per_round is not None and max_pairs_per_round > 0:
        depths = depths[:max_pairs_per_round]
        chosen = chosen[:max_pairs_per_round]
        rejected = rejected[:max_pairs_per_round]
        weights = weights[:max_pairs_per_round]

    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        depth_offsets = (depths.to(dtype=torch.float32) - 1.0).clamp(min=0.0)
        weights = weights * torch.exp(-depth_offsets / float(loss_decay_gamma))

    return (
        depths.to(device=device, dtype=torch.long) - 1,
        chosen.to(device=device, dtype=torch.long),
        rejected.to(device=device, dtype=torch.long),
        weights.to(device=device, dtype=torch.float32),
    )


def forward_dflash_round_hidden(
    *,
    draft_model: nn.Module,
    embed_tokens: nn.Module,
    round_item: Dict[str, Any],
    past_key_values: DynamicCache,
    block_size: int,
    mask_token_id: int,
    device: torch.device,
    detach_cache: bool,
) -> torch.Tensor:
    target_hidden = round_item["target_hidden"].to(device=device).unsqueeze(0)
    root_token_id = int(round_item["root_token_id"])
    decode_start = int(round_item["decode_start_index"])
    past_length = int(past_key_values.get_seq_length())

    block_ids = torch.full(
        (1, block_size),
        int(mask_token_id),
        dtype=torch.long,
        device=device,
    )
    block_ids[0, 0] = root_token_id
    noise_embedding = embed_tokens(block_ids)

    expected_position_count = target_hidden.shape[1] + block_size
    position_end = decode_start + block_size
    position_count = position_end - past_length
    if position_count != expected_position_count:
        position_end = past_length + expected_position_count
    position_ids = torch.arange(
        past_length,
        position_end,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    output_hidden = draft_model(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        is_causal=False,
    )
    draft_hidden = output_hidden[:, -(block_size - 1) :, :]
    past_key_values.crop(decode_start)
    if detach_cache:
        detach_dynamic_cache_(past_key_values)
    return draft_hidden


def gather_log_probs_from_hidden(
    *,
    hidden: torch.Tensor,
    lm_head: nn.Module,
    depth_indices: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    selected_hidden = hidden[0].index_select(0, depth_indices)
    logits = lm_head(selected_hidden).float()
    selected_logits = logits.gather(1, token_ids.view(-1, 1)).squeeze(1)
    return selected_logits - torch.logsumexp(logits, dim=-1)


def dpo_loss_from_logps(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = float(beta) * (policy_logratios - ref_logratios)

    if label_smoothing < 0.0 or label_smoothing >= 0.5:
        raise ValueError("--label-smoothing must be in [0, 0.5)")
    losses = -(
        (1.0 - label_smoothing) * F.logsigmoid(logits)
        + label_smoothing * F.logsigmoid(-logits)
    )
    if weights is not None:
        losses = losses * weights
        denom = weights.sum().clamp_min(1e-6)
    else:
        denom = torch.tensor(float(losses.numel()), device=losses.device)
    loss = losses.sum() / denom

    with torch.no_grad():
        metrics = {
            "pair_count": torch.tensor(float(logits.numel()), device=logits.device),
            "weighted_pair_count": denom.detach(),
            "margin": logits.mean() if logits.numel() else logits.new_zeros(()),
            "policy_logratio": policy_logratios.mean()
            if policy_logratios.numel()
            else logits.new_zeros(()),
            "ref_logratio": ref_logratios.mean()
            if ref_logratios.numel()
            else logits.new_zeros(()),
            "preference_accuracy": (logits > 0).float().mean()
            if logits.numel()
            else logits.new_zeros(()),
        }
    return loss, metrics


def compute_record_dpo_loss(
    *,
    record: Dict[str, Any],
    policy_model: nn.Module,
    ref_model: nn.Module,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    block_size: int,
    mask_token_id: int,
    device: torch.device,
    beta: float,
    label_smoothing: float,
    max_rounds_per_record: Optional[int],
    max_pairs_per_round: Optional[int],
    loss_decay_gamma: Optional[float],
    detach_draft_cache: bool,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    policy_cache = DynamicCache()
    ref_cache = DynamicCache()
    losses: List[torch.Tensor] = []
    metric_accumulator: Dict[str, float] = {
        "rounds": 0.0,
        "rounds_with_pairs": 0.0,
        "pairs": 0.0,
        "weighted_pairs": 0.0,
        "margin_sum": 0.0,
        "policy_logratio_sum": 0.0,
        "ref_logratio_sum": 0.0,
        "preference_accuracy_sum": 0.0,
    }

    rounds = record.get("rounds", [])
    if max_rounds_per_record is not None and max_rounds_per_record > 0:
        rounds = rounds[:max_rounds_per_record]

    for round_item in rounds:
        metric_accumulator["rounds"] += 1.0
        pair_tensors = _round_pair_tensors(
            round_item,
            device=device,
            max_pairs_per_round=max_pairs_per_round,
            loss_decay_gamma=loss_decay_gamma,
        )
        policy_hidden = forward_dflash_round_hidden(
            draft_model=policy_model,
            embed_tokens=embed_tokens,
            round_item=round_item,
            past_key_values=policy_cache,
            block_size=block_size,
            mask_token_id=mask_token_id,
            device=device,
            detach_cache=detach_draft_cache,
        )
        with torch.no_grad():
            ref_hidden = forward_dflash_round_hidden(
                draft_model=ref_model,
                embed_tokens=embed_tokens,
                round_item=round_item,
                past_key_values=ref_cache,
                block_size=block_size,
                mask_token_id=mask_token_id,
                device=device,
                detach_cache=True,
            )

        if pair_tensors is None:
            continue

        depth_indices, chosen_ids, rejected_ids, weights = pair_tensors
        policy_chosen = gather_log_probs_from_hidden(
            hidden=policy_hidden,
            lm_head=lm_head,
            depth_indices=depth_indices,
            token_ids=chosen_ids,
        )
        policy_rejected = gather_log_probs_from_hidden(
            hidden=policy_hidden,
            lm_head=lm_head,
            depth_indices=depth_indices,
            token_ids=rejected_ids,
        )
        with torch.no_grad():
            ref_chosen = gather_log_probs_from_hidden(
                hidden=ref_hidden,
                lm_head=lm_head,
                depth_indices=depth_indices,
                token_ids=chosen_ids,
            )
            ref_rejected = gather_log_probs_from_hidden(
                hidden=ref_hidden,
                lm_head=lm_head,
                depth_indices=depth_indices,
                token_ids=rejected_ids,
            )

        round_loss, round_metrics = dpo_loss_from_logps(
            policy_chosen_logps=policy_chosen,
            policy_rejected_logps=policy_rejected,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            beta=beta,
            label_smoothing=label_smoothing,
            weights=weights,
        )
        losses.append(round_loss * round_metrics["weighted_pair_count"])
        pair_count = float(round_metrics["pair_count"].item())
        weighted_count = float(round_metrics["weighted_pair_count"].item())
        metric_accumulator["rounds_with_pairs"] += 1.0
        metric_accumulator["pairs"] += pair_count
        metric_accumulator["weighted_pairs"] += weighted_count
        metric_accumulator["margin_sum"] += float(round_metrics["margin"].item())
        metric_accumulator["policy_logratio_sum"] += float(
            round_metrics["policy_logratio"].item()
        )
        metric_accumulator["ref_logratio_sum"] += float(
            round_metrics["ref_logratio"].item()
        )
        metric_accumulator["preference_accuracy_sum"] += float(
            round_metrics["preference_accuracy"].item()
        )

    weighted_pairs = metric_accumulator["weighted_pairs"]
    if not losses or weighted_pairs <= 0:
        return None, metric_accumulator
    loss = torch.stack(losses).sum() / max(weighted_pairs, 1e-6)
    rounds_with_pairs = max(metric_accumulator["rounds_with_pairs"], 1.0)
    metric_accumulator["margin"] = metric_accumulator["margin_sum"] / rounds_with_pairs
    metric_accumulator["policy_logratio"] = (
        metric_accumulator["policy_logratio_sum"] / rounds_with_pairs
    )
    metric_accumulator["ref_logratio"] = (
        metric_accumulator["ref_logratio_sum"] / rounds_with_pairs
    )
    metric_accumulator["preference_accuracy"] = (
        metric_accumulator["preference_accuracy_sum"] / rounds_with_pairs
    )
    return loss, metric_accumulator
