#!/usr/bin/env python3
# coding=utf-8
"""Export eval benchmark prompts as JSONL for DPO overfit probes."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_CACHE_DIR = ROOT_DIR / "cache" / "benchmark_sources"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
MTBENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe.  Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature.\n\nIf a "
    "question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the same benchmark prompts used by eval_dflash_acceptance.py "
            "to a prompt JSONL file consumable by prepare_dpo_data.py."
        )
    )
    parser.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        default=["mtbench:20", "gsm8k:20", "math500:20"],
        help="Format: <benchmark>:<num-samples>. Default exports 60 prompts total.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="cache/dataset/bench_test_sample.jsonl",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        default=None,
        help=(
            "Optional tokenizer path. Required only to render MT-Bench prompts with "
            "the exact target chat template used by HF eval."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


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


def download_text(url: str, cache_name: str) -> str:
    SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = SOURCE_CACHE_DIR / cache_name
    if not cache_path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            cache_path.write_bytes(response.read())
    return cache_path.read_text(encoding="utf-8")


def read_jsonl_from_url(url: str, cache_name: str) -> List[Dict[str, Any]]:
    text = download_text(url, cache_name)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def gsm8k_example(lines: List[Dict[str, Any]], index: int, include_answer: bool) -> str:
    result = "Question: " + lines[index]["question"] + "\nAnswer:"
    if include_answer:
        result += " " + lines[index]["answer"]
    return result


def gsm8k_few_shot_examples(lines: List[Dict[str, Any]], k: int = 5) -> str:
    return "".join(gsm8k_example(lines, index, True) + "\n\n" for index in range(k))


def gsm8k_answer_value(answer: str) -> Optional[int]:
    numbers = re.findall(r"\d+", answer.replace(",", ""))
    return int(numbers[-1]) if numbers else None


def extract_math_answer(output: str) -> Optional[str]:
    match = re.search(r"\\boxed\{([^}]+)\}", output)
    if match:
        return match.group(1).strip()
    match = re.search(r"\\boxed\s+([^\s]+)", output)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", output)
    return numbers[-1] if numbers else None


def load_benchmark_data(
    benchmark_name: str,
    num_samples: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, Any]]:
    if benchmark_name == "gsm8k":
        lines = read_jsonl_from_url(GSM8K_URL, "gsm8k_test.jsonl")
        few_shot = gsm8k_few_shot_examples(lines)
        questions = [
            {"question": gsm8k_example(lines, idx, False)}
            for idx in range(len(lines))
        ]
        labels = [gsm8k_answer_value(line["answer"]) for line in lines]
        extra = {"few_shot_examples": few_shot}
    elif benchmark_name == "math500":
        lines = read_jsonl_from_url(MATH500_URL, "math500_test.jsonl")
        questions = [{"question": item["problem"]} for item in lines]
        labels = [
            str(item["answer"]).strip()
            if "answer" in item
            else extract_math_answer(item.get("solution", ""))
            for item in lines
        ]
        extra = {}
    elif benchmark_name == "mtbench":
        lines = read_jsonl_from_url(MTBENCH_URL, "mtbench.jsonl")
        questions = [
            {"question_1": item["turns"][0], "question_2": item["turns"][1]}
            for item in lines
        ]
        labels = [None] * len(questions)
        extra = {}
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    if num_samples is not None:
        questions = questions[:num_samples]
        labels = labels[:num_samples]
    return questions, labels, extra


def render_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = [f"{message['role']}: {message['content']}\n" for message in messages]
    parts.append("assistant: ")
    return "".join(parts)


def get_stop_strings(benchmark_name: str) -> List[str]:
    if benchmark_name == "gsm8k":
        return ["Question", "Assistant:", "<|separator|>"]
    return []


def build_prompt(
    *,
    benchmark_name: str,
    question: Dict[str, Any],
    extra: Dict[str, Any],
    tokenizer,
) -> Optional[str]:
    if benchmark_name == "gsm8k":
        return extra["few_shot_examples"] + question["question"]
    if benchmark_name == "math500":
        return question["question"]
    if benchmark_name == "mtbench":
        if tokenizer is None:
            return None
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question["question_1"]})
        return render_chat_prompt(tokenizer, messages)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def build_conversations(
    benchmark_name: str,
    question: Dict[str, Any],
) -> List[Dict[str, str]]:
    if benchmark_name == "mtbench":
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question["question_1"]})
        return messages
    return [{"role": "user", "content": question["question"]}]


def main() -> None:
    args = parse_args()
    tokenizer = None
    if args.target_model_path is not None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required when --target-model-path is provided."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
        )

    records = []
    for benchmark_name, num_samples in parse_benchmark_specs(args.benchmark_list):
        questions, labels, extra = load_benchmark_data(benchmark_name, num_samples)
        for idx, (question, label) in enumerate(zip(questions, labels)):
            prompt = build_prompt(
                benchmark_name=benchmark_name,
                question=question,
                extra=extra,
                tokenizer=tokenizer,
            )
            record = {
                "id": f"{benchmark_name}:{idx}",
                "stop_strings": get_stop_strings(benchmark_name),
                "metadata": {
                    "benchmark": benchmark_name,
                    "benchmark_index": idx,
                    "label": label,
                    "question": question,
                },
            }
            if prompt is None:
                record["conversations"] = build_conversations(benchmark_name, question)
            else:
                record["prompt"] = prompt
            records.append(record)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} benchmark prompts to {output_path}")


if __name__ == "__main__":
    main()
