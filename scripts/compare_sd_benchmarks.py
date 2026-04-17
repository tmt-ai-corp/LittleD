#!/usr/bin/env python3
# coding=utf-8
"""Compare Eagle3, DFlash, and LittleBit-DFlash benchmark results."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare speculative decoding drafts")
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--benchmark-list", nargs="+", default=["mtbench:20", "gsm8k:20", "math500:20"])
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--eagle3-draft-model-path", type=str, default=None)
    parser.add_argument("--dflash-draft-model-path", type=str, default=None)
    parser.add_argument("--littlebit-draft-model-path", type=str, default=None)

    parser.add_argument("--eagle3-results-path", type=str, default=None)
    parser.add_argument("--dflash-results-path", type=str, default=None)
    parser.add_argument("--littlebit-results-path", type=str, default=None)

    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")

    parser.add_argument("--eagle3-port", type=int, default=30000)
    parser.add_argument("--eagle3-config-list", nargs="+", default=["1,3,1,4"])
    parser.add_argument("--eagle3-tp-size", type=int, default=1)
    parser.add_argument("--eagle3-mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--eagle3-attention-backend", type=str, default="fa3")
    parser.add_argument("--eagle3-timeout-for-server-launch", type=int, default=600)
    return parser.parse_args()


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def find_latest_result(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No result file found in {directory} with prefix {prefix}")
    return candidates[-1]


def maybe_run_eagle3(args) -> Path | None:
    if args.eagle3_results_path:
        return Path(args.eagle3_results_path)
    if not args.eagle3_draft_model_path:
        return None

    output_dir = Path(args.output_dir) / "eagle3"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "benchmarks/bench_eagle3.py",
        "--model-path",
        args.target_model_path,
        "--speculative-draft-model-path",
        args.eagle3_draft_model_path,
        "--port",
        str(args.eagle3_port),
        "--trust-remote-code",
        "--tp-size",
        str(args.eagle3_tp_size),
        "--mem-fraction-static",
        str(args.eagle3_mem_fraction_static),
        "--attention-backend",
        args.eagle3_attention_backend,
        "--timeout-for-server-launch",
        str(args.eagle3_timeout_for_server_launch),
        "--output-dir",
        str(output_dir),
        "--dtype",
        args.torch_dtype,
        "--name",
        "eagle3",
        "--benchmark-list",
        *args.benchmark_list,
        "--config-list",
        *args.eagle3_config_list,
    ]
    run_command(command)
    return find_latest_result(output_dir, "eagle3_results_")


def maybe_run_dflash_eval(
    *,
    args,
    draft_path: str | None,
    existing_path: str | None,
    draft_type: str,
    output_name: str,
) -> Path | None:
    if existing_path:
        return Path(existing_path)
    if not draft_path:
        return None

    output_path = Path(args.output_dir) / output_name
    command = [
        sys.executable,
        "scripts/eval_dflash_acceptance.py",
        "--target-model-path",
        args.target_model_path,
        "--draft-model-path",
        draft_path,
        "--draft-type",
        draft_type,
        "--benchmark-list",
        *args.benchmark_list,
        "--output-path",
        str(output_path),
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    run_command(command)
    return output_path


def parse_eagle3_results(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    parsed = {}
    for benchmark_name, runs in raw.items():
        if benchmark_name == "model":
            continue
        if not runs:
            continue
        metrics = runs[0]["metrics"]
        accept_length = sum(item["accept_length"] for item in metrics) / len(metrics)
        throughput = sum(item["output_throughput"] for item in metrics) / len(metrics)
        accuracy_values = [item.get("accuracy") for item in metrics if item.get("accuracy") is not None]
        parsed[benchmark_name] = {
            "accept_length": accept_length,
            "output_throughput": throughput,
            "accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else None,
            "source": str(path),
            "backend": "sglang",
        }
    return parsed


def parse_dflash_results(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        item["benchmark"]: {
            "accept_length": item["accept_length"],
            "output_throughput": item["output_throughput"],
            "accuracy": item["accuracy"],
            "source": str(path),
            "backend": "hf_reference",
        }
        for item in raw["benchmarks"]
    }


def make_markdown_table(summary: dict) -> str:
    lines = [
        "| Benchmark | Variant | Accept Length | Throughput | Accuracy | Backend |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for benchmark_name, variants in summary.items():
        ranked = sorted(
            variants.items(),
            key=lambda item: item[1]["accept_length"],
            reverse=True,
        )
        for variant_name, metrics in ranked:
            accuracy = (
                f"{metrics['accuracy']:.4f}"
                if metrics["accuracy"] is not None
                else "-"
            )
            lines.append(
                f"| {benchmark_name} | {variant_name} | {metrics['accept_length']:.3f} | "
                f"{metrics['output_throughput']:.3f} | {accuracy} | {metrics['backend']} |"
            )
    return "\n".join(lines)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eagle3_path = maybe_run_eagle3(args)
    dflash_path = maybe_run_dflash_eval(
        args=args,
        draft_path=args.dflash_draft_model_path,
        existing_path=args.dflash_results_path,
        draft_type="dflash",
        output_name="dflash_acceptance.json",
    )
    littlebit_path = maybe_run_dflash_eval(
        args=args,
        draft_path=args.littlebit_draft_model_path,
        existing_path=args.littlebit_results_path,
        draft_type="littlebit_dflash",
        output_name="littlebit_dflash_acceptance.json",
    )

    variant_results = {}
    if eagle3_path:
        variant_results["eagle3"] = parse_eagle3_results(eagle3_path)
    if dflash_path:
        variant_results["dflash"] = parse_dflash_results(dflash_path)
    if littlebit_path:
        variant_results["littlebit_dflash"] = parse_dflash_results(littlebit_path)

    summary = {}
    for variant_name, benchmarks in variant_results.items():
        for benchmark_name, metrics in benchmarks.items():
            summary.setdefault(benchmark_name, {})[variant_name] = metrics

    markdown = make_markdown_table(summary)
    comparison = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_model_path": args.target_model_path,
        "summary": summary,
        "notes": [
            "Acceptance length is the primary comparison metric.",
            "Eagle3 throughput comes from SGLang serving benchmarks.",
            "DFlash and LittleBit-DFlash throughput come from HF reference evaluation and are not deployment-kernel numbers.",
        ],
        "artifacts": {
            "eagle3": str(eagle3_path) if eagle3_path else None,
            "dflash": str(dflash_path) if dflash_path else None,
            "littlebit_dflash": str(littlebit_path) if littlebit_path else None,
        },
        "markdown_table": markdown,
    }

    summary_path = output_dir / "benchmark_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)

    markdown_path = output_dir / "benchmark_comparison.md"
    markdown_path.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print(f"Saved comparison to {summary_path}")


if __name__ == "__main__":
    main()
