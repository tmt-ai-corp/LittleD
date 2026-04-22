#!/usr/bin/env python3
# coding=utf-8
"""Prepare offline DDTree preference data for LittleBit-DFlash DPO."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

try:
    from benchmarks.benchmarker import BENCHMARKS
    from benchmarks.benchmarker.mtbench import SYSTEM_PROMPT
except ImportError:
    BENCHMARKS = {}
    SYSTEM_PROMPT = ""

from specforge.core.dflash_dpo import (
    DPO_DATA_VERSION,
    build_position_preference_pairs,
    compact_tree_summary,
    count_record_pairs,
    count_record_rounds,
    resolve_dtype,
)
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


@dataclass
class PromptItem:
    sample_id: str
    source: str
    prompt: str
    stop_strings: List[str]
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect DDTree accepted-vs-sibling preference pairs for DFlash DPO."
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument("--draft-model-path", type=str, required=True)
    model_group.add_argument(
        "--draft-type",
        type=str,
        default="littlebit_dflash",
        choices=["dflash", "littlebit_dflash"],
    )
    model_group.add_argument("--device", type=str, default="cuda")
    model_group.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    model_group.add_argument(
        "--target-attention-backend",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Target attention backend. DDTree verification needs a backend that accepts a 4D tree mask.",
    )
    model_group.add_argument(
        "--draft-attention-backend",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    model_group.add_argument("--trust-remote-code", action="store_true")

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Optional JSON/JSONL prompt data. Records may contain prompt/text/question/messages/conversations.",
    )
    data_group.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl"],
    )
    data_group.add_argument("--prompt-field", type=str, default="prompt")
    data_group.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        default=None,
        help="Optional eval-style benchmark specs, e.g. gsm8k:100 math500:100 mtbench:50.",
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional total cap across input and benchmark sources.",
    )
    data_group.add_argument(
        "--stop-strings",
        type=str,
        nargs="*",
        default=[],
        help="Stop strings used for --input-path prompts.",
    )
    data_group.add_argument("--max-input-tokens", type=int, default=2048)
    data_group.add_argument("--max-new-tokens", type=int, default=256)
    data_group.add_argument("--temperature", type=float, default=0.0)

    tree_group = parser.add_argument_group("ddtree dpo")
    tree_group.add_argument("--ddtree-size", type=int, default=32)
    tree_group.add_argument(
        "--include-terminal-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compare the target fallback next token against children where the accepted path stops.",
    )
    tree_group.add_argument(
        "--max-pairs-per-round",
        type=int,
        default=None,
        help="Optional cap on pairs stored per round.",
    )
    tree_group.add_argument(
        "--hidden-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="CPU dtype used for saved target_hidden features.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--samples-per-shard", type=int, default=32)

    quant_group = parser.add_argument_group("littlebit fallback config")
    quant_group.add_argument("--quant-func", type=str, default="STEBinary")
    quant_group.add_argument("--split-dim", type=int, default=1024)
    quant_group.add_argument("--eff-bit", type=float, default=0.5)
    quant_group.add_argument("--kv-factor", type=float, default=1.0)
    quant_group.add_argument("--min-split-dim", type=int, default=8)
    quant_group.add_argument("--residual", action="store_true")

    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def read_json_records(path: Path, input_format: str) -> List[Any]:
    fmt = input_format
    if fmt == "auto":
        fmt = "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "json"
    if fmt == "jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    for key in ("data", "train", "samples", "records"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return payload[key]
    return [payload]


def parse_benchmark_specs(items: List[str]) -> List[tuple[str, Optional[int]]]:
    specs = []
    for item in items:
        splits = item.split(":")
        if len(splits) == 1:
            specs.append((splits[0], None))
        elif len(splits) == 2:
            specs.append((splits[0], int(splits[1])))
        else:
            raise ValueError(f"Invalid benchmark spec: {item}")
    return specs


def build_stop_token_sequences(tokenizer, stop_strings: List[str]) -> List[List[int]]:
    stop_token_sequences = []
    seen = set()
    for stop_string in stop_strings:
        for prefix in ("", "\n", "\n\n", " "):
            token_ids = tokenizer.encode(
                prefix + stop_string,
                add_special_tokens=False,
            )
            if not token_ids:
                continue
            key = tuple(token_ids)
            if key not in seen:
                seen.add(key)
                stop_token_sequences.append(token_ids)
    return stop_token_sequences


def truncate_at_stop_strings(text: str, stop_strings: List[str]) -> str:
    stop_positions = [text.find(stop) for stop in stop_strings if stop in text]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


def get_stop_strings(benchmark_name: str) -> List[str]:
    if benchmark_name == "gsm8k":
        return ["Question", "Assistant:", "<|separator|>"]
    return []


def render_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = [f"{message['role']}: {message['content']}\n" for message in messages]
    parts.append("assistant: ")
    return "".join(parts)


def build_benchmark_prompt(
    benchmark_name: str,
    benchmarker,
    question: Dict[str, Any],
    tokenizer,
) -> str:
    if benchmark_name == "gsm8k":
        return benchmarker.few_shot_examples + question["question"]
    if benchmark_name == "math500":
        return question["question"]
    if benchmark_name == "mtbench":
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question["question_1"]})
        return render_chat_prompt(tokenizer, messages)
    raise ValueError(f"Unsupported benchmark for DPO data: {benchmark_name}")


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for message in messages:
        role = message.get("role", message.get("from", "user"))
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "model"}:
            role = "assistant"
        elif role == "system":
            role = "system"
        content = message.get("content", message.get("value", ""))
        normalized.append({"role": role, "content": str(content)})
    return normalized


def render_record_prompt(record: Any, tokenizer, prompt_field: str) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return str(record)
    if prompt_field in record:
        return str(record[prompt_field])
    for key in ("prompt", "text", "question", "instruction"):
        if key in record:
            return str(record[key])
    messages = record.get("messages", record.get("conversations"))
    if isinstance(messages, list):
        normalized = normalize_messages(messages)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(f"{item['role']}: {item['content']}" for item in normalized)
    raise ValueError(f"Could not infer a prompt from record keys: {sorted(record)}")


def get_record_stop_strings(record: Any, default_stop_strings: List[str]) -> List[str]:
    if not isinstance(record, dict):
        return list(default_stop_strings)
    for key in ("stop_strings", "stop", "stop_sequences"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        raise ValueError(f"Expected {key} to be a string or list, got {type(value)}")
    return list(default_stop_strings)


def get_record_metadata(record: Any, idx: int) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"input_index": idx}
    if not isinstance(record, dict):
        return metadata
    if isinstance(record.get("metadata"), dict):
        metadata.update(record["metadata"])
    for key in ("id", "benchmark", "benchmark_index", "source"):
        if key in record:
            metadata[key] = record[key]
    return metadata


def iter_prompt_items(args, tokenizer) -> Iterable[PromptItem]:
    emitted = 0

    if args.input_path:
        path = Path(args.input_path)
        for idx, record in enumerate(read_json_records(path, args.input_format)):
            prompt = render_record_prompt(record, tokenizer, args.prompt_field)
            stop_strings = get_record_stop_strings(record, args.stop_strings)
            yield PromptItem(
                sample_id=f"input:{idx}",
                source=str(path),
                prompt=prompt,
                stop_strings=stop_strings,
                metadata=get_record_metadata(record, idx),
            )
            emitted += 1
            if args.num_samples is not None and emitted >= args.num_samples:
                return

    if args.benchmark_list:
        if not BENCHMARKS:
            raise ImportError("benchmarks package is unavailable but --benchmark-list was set.")
        for benchmark_name, num_samples in parse_benchmark_specs(args.benchmark_list):
            benchmarker_cls = BENCHMARKS.get(benchmark_name)
            if benchmarker_cls is None:
                raise ValueError(f"Unknown benchmark: {benchmark_name}")
            benchmarker = benchmarker_cls(num_samples=num_samples)
            questions, labels = benchmarker.load_data()
            stop_strings = get_stop_strings(benchmark_name)
            for idx, (question, label) in enumerate(zip(questions, labels)):
                prompt = build_benchmark_prompt(
                    benchmark_name,
                    benchmarker,
                    question,
                    tokenizer,
                )
                yield PromptItem(
                    sample_id=f"{benchmark_name}:{idx}",
                    source=f"benchmark:{benchmark_name}",
                    prompt=prompt,
                    stop_strings=stop_strings,
                    metadata={
                        "benchmark": benchmark_name,
                        "benchmark_index": idx,
                        "label": label,
                    },
                )
                emitted += 1
                if args.num_samples is not None and emitted >= args.num_samples:
                    return

    if emitted == 0:
        raise ValueError("Provide --input-path or --benchmark-list.")


def resolve_mask_token_id(tokenizer, draft_model) -> int:
    mask_token_id = getattr(draft_model, "mask_token_id", None)
    if mask_token_id is None:
        dflash_config = getattr(draft_model.config, "dflash_config", {}) or {}
        mask_token_id = dflash_config.get("mask_token_id")
    if mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError(
            "Could not resolve mask_token_id from the DFlash checkpoint or tokenizer."
        )
    draft_model.mask_token_id = int(mask_token_id)
    return int(mask_token_id)


def load_models(args, *, device: torch.device, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.target_attention_backend,
    ).to(device)
    target_model.eval()

    if args.draft_type == "littlebit_dflash":
        draft_model = load_quantized_dflash_model(
            args.draft_model_path,
            device=device,
            torch_dtype=torch_dtype,
            quant_args=args,
            do_train=False,
        )
    else:
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        draft_model.eval()
    draft_model.config._attn_implementation = args.draft_attention_backend
    draft_model.eval()
    return tokenizer, target_model, draft_model


@torch.inference_mode()
def collect_dpo_record(
    *,
    item: PromptItem,
    tokenizer,
    target_model,
    draft_model,
    args,
    device: torch.device,
    save_hidden_dtype: torch.dtype,
    mask_token_id: int,
) -> Dict[str, Any]:
    encoded = tokenizer(
        item.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    num_input_tokens = int(input_ids.shape[1])
    max_new_tokens = int(args.max_new_tokens)
    max_length = num_input_tokens + max_new_tokens
    block_size = int(draft_model.block_size)
    draft_horizon = block_size - 1
    tree_budget = max(int(args.ddtree_size), 0)
    max_tree_nodes = 1 + tree_budget
    target_dtype = getattr(target_model, "dtype", next(target_model.parameters()).dtype)

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    verify_input_ids_buffer = torch.empty(
        (1, max_tree_nodes),
        dtype=torch.long,
        device=device,
    )
    verify_position_ids_buffer = torch.empty(
        (1, max_tree_nodes),
        dtype=torch.long,
        device=device,
    )
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target_dtype,
        device=device,
    )
    tree_visibility_buffer = torch.empty(
        (max_tree_nodes, max_tree_nodes),
        dtype=torch.bool,
        device=device,
    )

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = time.perf_counter()
    output = target_model(
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
        args.temperature,
    )
    target_hidden = extract_context_feature(
        output.hidden_states,
        draft_model.target_layer_ids,
    )

    stop_token_ids = (
        [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    )
    stop_token_sequences = build_stop_token_sequences(tokenizer, item.stop_strings)
    stop_sequence_start = None
    previous_tree_start = 0
    previous_tree_length = 0
    start = num_input_tokens
    round_index = 0
    rounds: List[Dict[str, Any]] = []
    acceptance_lengths: List[int] = []
    ddtree_stage_times = {
        "draft": 0.0,
        "tree_build": 0.0,
        "tree_compile": 0.0,
        "verify": 0.0,
        "commit": 0.0,
    }

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]
        round_target_hidden = target_hidden.detach().to(
            device="cpu",
            dtype=save_hidden_dtype,
        )[0].contiguous()

        stage_start = time.perf_counter()
        noise_embedding = target_model.model.embed_tokens(block_output_ids)
        draft_hidden = draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        draft_logits = target_model.lm_head(draft_hidden[:, -draft_horizon:, :])
        past_key_values_draft.crop(start)
        ddtree_stage_times["draft"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        node_token_ids, node_depths, parents, child_maps, visibility_cpu = (
            build_ddtree_tree(draft_logits[0], tree_budget)
        )
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
        output = target_model(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        ddtree_stage_times["verify"] += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        posterior = sample(output.logits, args.temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(
            accepted_indices,
            dtype=torch.long,
            device=device,
        )
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)
        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = int(next_token)

        pairs = build_position_preference_pairs(
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            child_maps=child_maps,
            accepted_indices=accepted_indices,
            next_token=int(next_token),
            block_size=block_size,
            include_terminal_pairs=args.include_terminal_pairs,
            max_pairs_per_round=args.max_pairs_per_round,
        )
        tree = compact_tree_summary(
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            parents=parents,
            accepted_indices=accepted_indices,
            next_token=int(next_token),
            posterior=posterior,
            draft_logits=draft_logits[0],
        )
        rounds.append(
            {
                "round_index": int(round_index),
                "decode_start_index": int(start),
                "root_token_id": int(root_token[0, 0].item()),
                "target_hidden": round_target_hidden,
                "pairs": pairs,
                "tree": tree,
                "verify_input_ids": verify_input_ids.detach().cpu().to(torch.long)[0],
                "verify_position_ids": verify_position_ids.detach()
                .cpu()
                .to(torch.long)[0],
                "accepted_token_ids": accepted_tokens.detach().cpu().to(torch.long)[0],
            }
        )

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        tree_target_hidden = extract_context_feature(
            output.hidden_states,
            draft_model.target_layer_ids,
        )
        target_hidden = tree_target_hidden.index_select(1, accepted_index_tensor)
        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        ddtree_stage_times["commit"] += time.perf_counter() - stage_start

        search_end = min(start + 1, max_length, output_ids.shape[1])
        stop_sequence_index = find_first_stop_sequence(
            output_ids[0, num_input_tokens:search_end],
            stop_token_sequences,
        )
        if stop_sequence_index is not None:
            stop_sequence_start = num_input_tokens + stop_sequence_index
            break

        if stop_token_ids is not None:
            stop_ids = torch.tensor(stop_token_ids, device=device)
            if torch.isin(output_ids[0, num_input_tokens:search_end], stop_ids).any():
                break

        round_index += 1

    final_ids = output_ids[:, :max_length]
    if stop_sequence_start is not None:
        final_ids = final_ids[:, :stop_sequence_start]
    final_ids = final_ids[:, final_ids[0] != mask_token_id]
    if stop_sequence_start is None and stop_token_ids is not None:
        stop_token_ids_tensor = torch.tensor(stop_token_ids, device=device)
        stop_token_indices = torch.isin(
            final_ids[0][num_input_tokens:],
            stop_token_ids_tensor,
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            final_ids = final_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    generated_text = tokenizer.decode(
        final_ids[0][num_input_tokens:],
        skip_special_tokens=True,
    )
    generated_text = truncate_at_stop_strings(generated_text, item.stop_strings)

    return {
        "version": DPO_DATA_VERSION,
        "sample_id": item.sample_id,
        "source": item.source,
        "metadata": item.metadata,
        "prompt_text": item.prompt,
        "prompt_input_ids": input_ids.detach().cpu().to(torch.long)[0],
        "generated_token_ids": final_ids.detach().cpu().to(torch.long)[
            0, num_input_tokens:
        ],
        "generated_text": generated_text,
        "num_input_tokens": num_input_tokens,
        "num_new_tokens": int(max(final_ids.shape[1] - num_input_tokens, 0)),
        "num_speculation_steps": len(acceptance_lengths),
        "acceptance_lengths": acceptance_lengths,
        "accept_length": (
            float(sum(acceptance_lengths)) / len(acceptance_lengths)
            if acceptance_lengths
            else 1.0
        ),
        "time_to_first_token": time_to_first_token,
        "ddtree_stage_times": ddtree_stage_times,
        "rounds": rounds,
    }


class ShardWriter:
    def __init__(self, output_dir: Path, samples_per_shard: int):
        self.output_dir = output_dir
        self.samples_per_shard = max(int(samples_per_shard), 1)
        self.records: List[Dict[str, Any]] = []
        self.shards: List[Dict[str, Any]] = []
        self.shard_index = 0

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        if len(self.records) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        shard_name = f"shard_{self.shard_index:05d}.pt"
        shard_path = self.output_dir / shard_name
        num_rounds = sum(count_record_rounds(record) for record in self.records)
        num_pairs = sum(count_record_pairs(record) for record in self.records)
        torch.save(
            {"version": DPO_DATA_VERSION, "records": self.records},
            shard_path,
        )
        self.shards.append(
            {
                "path": shard_name,
                "num_records": len(self.records),
                "num_rounds": num_rounds,
                "num_pairs": num_pairs,
            }
        )
        self.records = []
        self.shard_index += 1


def main() -> None:
    args = parse_args()
    torch_dtype = resolve_dtype(args.torch_dtype)
    save_hidden_dtype = resolve_dtype(args.hidden_dtype)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, target_model, draft_model = load_models(
        args,
        device=device,
        torch_dtype=torch_dtype,
    )
    mask_token_id = resolve_mask_token_id(tokenizer, draft_model)

    writer = ShardWriter(output_dir, args.samples_per_shard)
    started_at = time.time()
    total_records = 0
    total_rounds = 0
    total_pairs = 0
    skipped = 0

    prompt_items = list(iter_prompt_items(args, tokenizer))
    for item in tqdm(prompt_items, desc="Collecting DDTree DPO data"):
        try:
            record = collect_dpo_record(
                item=item,
                tokenizer=tokenizer,
                target_model=target_model,
                draft_model=draft_model,
                args=args,
                device=device,
                save_hidden_dtype=save_hidden_dtype,
                mask_token_id=mask_token_id,
            )
        except Exception as exc:
            skipped += 1
            print(f"WARNING: skipped {item.sample_id}: {exc}")
            continue
        record_pairs = count_record_pairs(record)
        writer.add(record)
        total_records += 1
        total_rounds += count_record_rounds(record)
        total_pairs += record_pairs

    writer.flush()
    manifest = {
        "version": DPO_DATA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started_at,
        "target_model_path": args.target_model_path,
        "draft_model_path": args.draft_model_path,
        "draft_type": args.draft_type,
        "torch_dtype": args.torch_dtype,
        "hidden_dtype": args.hidden_dtype,
        "ddtree_size": args.ddtree_size,
        "include_terminal_pairs": args.include_terminal_pairs,
        "max_pairs_per_round": args.max_pairs_per_round,
        "block_size": int(draft_model.block_size),
        "mask_token_id": int(mask_token_id),
        "target_layer_ids": [int(item) for item in draft_model.target_layer_ids],
        "num_records": total_records,
        "num_skipped": skipped,
        "num_rounds": total_rounds,
        "num_pairs": total_pairs,
        "shards": writer.shards,
        "args": vars(args),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(
        f"Saved DDTree DPO data to {output_dir} "
        f"({total_records} records, {total_rounds} rounds, {total_pairs} pairs, "
        f"{skipped} skipped)"
    )


if __name__ == "__main__":
    main()
