#!/usr/bin/env python3
# coding=utf-8
"""Evaluate DFlash-family draft models with acceptance-length-first metrics."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.benchmarker import BENCHMARKS
from benchmarks.benchmarker.gsm8k import GSM8KBenchmarker
from benchmarks.benchmarker.math500 import Math500Benchmarker
from benchmarks.benchmarker.mtbench import MTBenchBenchmarker, SYSTEM_PROMPT
from specforge.littlebit import load_quantized_dflash_model
from specforge.modeling.draft.dflash import DFlashDraftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DFlash acceptance length")
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
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Optional overrides for quantized checkpoints without littlebit_config.json.
    parser.add_argument("--quant-func", type=str, default="STEBinary")
    parser.add_argument("--split-dim", type=int, default=1024)
    parser.add_argument("--eff-bit", type=float, default=0.5)
    parser.add_argument("--kv-factor", type=float, default=1.0)
    parser.add_argument("--min-split-dim", type=int, default=8)
    parser.add_argument("--residual", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_benchmark_specs(items: List[str]) -> List[Tuple[str, Optional[int]]]:
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


def build_prompt(
    benchmark_name: str,
    benchmarker,
    question: Dict[str, Any],
    tokenizer,
    generated_answers: Optional[List[str]] = None,
) -> str:
    generated_answers = generated_answers or []
    if benchmark_name == "gsm8k":
        return benchmarker.few_shot_examples + question["question"]
    if benchmark_name == "math500":
        return question["question"]
    if benchmark_name == "mtbench":
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question["question_1"]})
        if generated_answers:
            messages.append({"role": "assistant", "content": generated_answers[0]})
            messages.append({"role": "user", "content": question["question_2"]})
        return render_chat_prompt(tokenizer, messages)
    raise ValueError(f"Unsupported benchmark for HF DFlash evaluation: {benchmark_name}")


def render_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
        is not None
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts = []
    for message in messages:
        parts.append(f"{message['role']}: {message['content']}\n")
    parts.append("assistant: ")
    return "".join(parts)


def get_stop_strings(benchmark_name: str) -> List[str]:
    if benchmark_name == "gsm8k":
        # Match the official SGLang GSM8K benchmark stop conditions. Without
        # these, few-shot prompts often continue into the next "Question",
        # inflating acceptance length and breaking numeric answer extraction.
        return ["Question", "Assistant:", "<|separator|>"]
    return []


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


def run_generation(
    *,
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
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
    output_ids, stats = draft_model.spec_generate(
        target=target_model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        return_stats=True,
        stop_token_sequences=stop_token_sequences,
    )
    generated = tokenizer.decode(
        output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
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

    for question, label in zip(questions, labels):
        turn_outputs = []
        turn_stats = []
        stop_strings = get_stop_strings(benchmark_name)
        start = time.perf_counter()

        if benchmark_name == "mtbench":
            for turn_idx in range(2):
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
            )
            turn_outputs.append(generated)
            turn_stats.append(stats)

        elapsed = time.perf_counter() - start
        total_latency += elapsed

        prompt_new_tokens = sum(item["num_new_tokens"] for item in turn_stats)
        prompt_steps = sum(item["num_speculation_steps"] for item in turn_stats)
        total_new_tokens += prompt_new_tokens
        total_spec_steps += prompt_steps

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
        "prompts": prompt_results,
        "note": "throughput is a HF reference number; acceptance length is the primary metric for LittleBit-DFlash.",
    }


def main():
    args = parse_args()
    torch_dtype = resolve_dtype(args.torch_dtype)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
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
        "target_model_path": args.target_model_path,
        "draft_model_path": args.draft_model_path,
        "draft_type": args.draft_type,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
            f"{benchmark_name}: accept_length={benchmark_result['accept_length']:.3f}, "
            f"throughput={benchmark_result['output_throughput']:.3f}, "
            f"accuracy={benchmark_result['accuracy']}"
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
