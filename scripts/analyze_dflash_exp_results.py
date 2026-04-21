#!/usr/bin/env python3
# coding=utf-8
"""Summarize large eval_dflash_acceptance_exp.py JSON outputs.

The experimental eval can easily write hundreds of MB because option 1/2/3 add
per-round tree paths, leaf-redraft batches, and tensor-stat snapshots. This
script turns that raw JSON into a small reading report plus an optional compact
JSON summary.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


OPTION_LABELS = {
    1: "DDTree structure stats",
    2: "Leaf-batch DFlash redraft",
    3: "Hidden/logit tensor stats",
}


class NumberSeries:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: Any) -> None:
        if value is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if math.isnan(number) or math.isinf(number):
            return
        self.values.append(number)

    def extend(self, values: Iterable[Any]) -> None:
        for value in values:
            self.add(value)

    def summary(self) -> dict[str, Any]:
        if not self.values:
            return {"count": 0}
        values = sorted(self.values)
        count = len(values)
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count

        def percentile(pct: float) -> float:
            if count == 1:
                return values[0]
            rank = (count - 1) * pct
            low = math.floor(rank)
            high = math.ceil(rank)
            if low == high:
                return values[low]
            weight = rank - low
            return values[low] * (1.0 - weight) + values[high] * weight

        return {
            "count": count,
            "min": values[0],
            "mean": mean,
            "std": math.sqrt(variance),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": values[-1],
        }


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "-"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.{digits}f}"


def fmt_percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def get_path(obj: dict[str, Any], path: str, default=None):
    current: Any = obj
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def add_histogram(counter: Counter, histogram: Optional[dict[str, Any]]) -> None:
    if not isinstance(histogram, dict):
        return
    for key, value in histogram.items():
        try:
            counter[str(key)] += int(value)
        except (TypeError, ValueError):
            continue


def format_histogram(counter: Counter, *, top_k: int) -> str:
    if not counter:
        return "-"

    def sort_key(item):
        key, count = item
        try:
            numeric_key = int(key)
        except ValueError:
            numeric_key = key
        return (-count, numeric_key)

    rows = sorted(counter.items(), key=sort_key)[:top_k]
    return ", ".join(f"{key}:{count}" for key, count in rows)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def stat_row(name: str, summary: dict[str, Any]) -> list[str]:
    return [
        name,
        str(summary.get("count", 0)),
        fmt_num(summary.get("mean")),
        fmt_num(summary.get("p50")),
        fmt_num(summary.get("p95")),
        fmt_num(summary.get("max")),
    ]


def brief_options(options: Any) -> str:
    if not options:
        return "none"
    labels = []
    for option in options:
        try:
            option_int = int(option)
        except (TypeError, ValueError):
            labels.append(str(option))
            continue
        labels.append(f"{option_int} ({OPTION_LABELS.get(option_int, 'unknown')})")
    return ", ".join(labels)


class ExpAnalyzer:
    def __init__(self, *, top_rounds: int, top_hist: int) -> None:
        self.top_rounds = top_rounds
        self.top_hist = top_hist
        self.benchmarks: list[dict[str, Any]] = []

        self.acceptance = defaultdict(NumberSeries)
        self.stage_times = defaultdict(NumberSeries)
        self.tree = defaultdict(NumberSeries)
        self.tree_histograms = defaultdict(Counter)
        self.leaf = defaultdict(NumberSeries)
        self.leaf_histograms = defaultdict(Counter)
        self.tensor = defaultdict(NumberSeries)

        self.accepted_ended_at_leaf = Counter()
        self.leaf_sampled_first_tokens = Counter()
        self.interesting_rounds: list[dict[str, Any]] = []
        self.total_prompts = 0
        self.total_turns = 0
        self.total_rounds = 0
        self.total_leaf_groups = 0

    def analyze_benchmark(self, benchmark: dict[str, Any]) -> None:
        bench_summary = {
            "benchmark": benchmark.get("benchmark"),
            "num_samples": benchmark.get("num_samples"),
            "latency": benchmark.get("latency"),
            "num_new_tokens": benchmark.get("num_new_tokens"),
            "num_speculation_steps": benchmark.get("num_speculation_steps"),
            "accept_length": benchmark.get("accept_length"),
            "output_throughput": benchmark.get("output_throughput"),
            "accuracy": benchmark.get("accuracy"),
            "generation_mode": benchmark.get("generation_mode"),
            "ddtree_size": benchmark.get("ddtree_size"),
            "exp_options": benchmark.get("exp_options"),
        }
        self.benchmarks.append(bench_summary)

        benchmark_name = str(benchmark.get("benchmark", "unknown"))
        for prompt_index, prompt in enumerate(benchmark.get("prompts", [])):
            self.total_prompts += 1
            self.acceptance["prompt.accept_length"].add(prompt.get("accept_length"))
            self.acceptance["prompt.num_new_tokens"].add(prompt.get("num_new_tokens"))
            self.acceptance["prompt.num_speculation_steps"].add(
                prompt.get("num_speculation_steps")
            )
            self.acceptance["prompt.latency"].add(prompt.get("latency"))

            for turn_index, stats in enumerate(prompt.get("turn_stats", [])):
                self.total_turns += 1
                self.acceptance["turn.accept_length"].add(stats.get("accept_length"))
                self.acceptance["turn.num_new_tokens"].add(stats.get("num_new_tokens"))
                self.acceptance["turn.num_speculation_steps"].add(
                    stats.get("num_speculation_steps")
                )
                self.acceptance["turn.time_to_first_token"].add(
                    stats.get("time_to_first_token")
                )
                self.acceptance["round.acceptance_length"].extend(
                    stats.get("acceptance_lengths", [])
                )

                for stage_name, elapsed in (stats.get("ddtree_stage_times") or {}).items():
                    self.stage_times[stage_name].add(elapsed)

                self._record_tensor_stats(
                    "prefill.target_logits",
                    get_path(stats, "exp_prefill.target_logits"),
                )
                self._record_hidden_stack(
                    "prefill.target_hidden_states",
                    get_path(stats, "exp_prefill.target_hidden_states"),
                )
                self._record_tensor_stats(
                    "prefill.dflash_context_feature",
                    get_path(stats, "exp_prefill.dflash_context_feature"),
                )

                for round_stats in stats.get("exp_rounds", []):
                    self._analyze_round(
                        benchmark_name=benchmark_name,
                        prompt_index=prompt_index,
                        turn_index=turn_index,
                        round_stats=round_stats,
                    )

    def _analyze_round(
        self,
        *,
        benchmark_name: str,
        prompt_index: int,
        turn_index: int,
        round_stats: dict[str, Any],
    ) -> None:
        self.total_rounds += 1
        tree = round_stats.get("tree") or {}
        if tree.get("enabled") is not False:
            self._record_tree(tree)
            self._record_interesting_round(
                benchmark_name=benchmark_name,
                prompt_index=prompt_index,
                turn_index=turn_index,
                tree=tree,
            )

        self._record_tensor_stats(
            "round.draft.noise_embedding",
            get_path(round_stats, "draft.noise_embedding"),
        )
        self._record_tensor_stats(
            "round.draft.draft_hidden",
            get_path(round_stats, "draft.draft_hidden"),
        )
        self._record_hidden_stack(
            "round.draft.draft_hidden_states",
            get_path(round_stats, "draft.draft_hidden_states"),
        )
        self._record_tensor_stats(
            "round.draft.draft_logits",
            get_path(round_stats, "draft.draft_logits"),
        )
        self._record_tensor_stats(
            "round.verify.attention_mask",
            get_path(round_stats, "verify.attention_mask"),
        )
        self._record_tensor_stats(
            "round.verify.target_logits",
            get_path(round_stats, "verify.target_logits"),
        )
        self._record_hidden_stack(
            "round.verify.target_hidden_states",
            get_path(round_stats, "verify.target_hidden_states"),
        )
        self._record_tensor_stats(
            "round.verify.tree_target_hidden",
            get_path(round_stats, "verify.tree_target_hidden"),
        )
        self._record_tensor_stats(
            "round.commit.next_context_feature",
            get_path(round_stats, "commit.next_context_feature"),
        )

        leaf_redraft = round_stats.get("leaf_redraft") or {}
        if leaf_redraft.get("enabled"):
            self._record_leaf_redraft(leaf_redraft)

    def _record_tree(self, tree: dict[str, Any]) -> None:
        scalar_keys = [
            "node_count_with_root",
            "non_root_node_count",
            "max_depth",
            "max_width",
            "mean_width",
            "leaf_count",
            "leaf_fraction",
            "internal_node_count",
            "accepted_path_length_with_root",
            "accepted_draft_token_count",
            "accepted_path_depth",
            "visibility_true_count",
            "visibility_density",
        ]
        for key in scalar_keys:
            self.tree[key].add(tree.get(key))

        for key in [
            "non_root_depth",
            "leaf_depth",
            "branching_factor_all_nodes",
            "branching_factor_internal_nodes",
            "visibility_row_true_count",
        ]:
            self._record_summary_fields(f"tree.{key}", tree.get(key))

        probability = tree.get("probability") or {}
        for key in [
            "local_logprob",
            "local_prob",
            "cumulative_logprob",
            "cumulative_prob",
            "leaf_cumulative_logprob",
            "leaf_cumulative_prob",
            "node_rank",
        ]:
            self._record_summary_fields(f"tree.probability.{key}", probability.get(key))

        add_histogram(self.tree_histograms["depth"], tree.get("depth_histogram"))
        add_histogram(
            self.tree_histograms["leaf_depth"],
            tree.get("leaf_depth_histogram"),
        )
        add_histogram(
            self.tree_histograms["branching_factor"],
            tree.get("branching_factor_histogram"),
        )
        add_histogram(
            self.tree_histograms["width_by_depth_with_root"],
            tree.get("width_by_depth_with_root"),
        )
        add_histogram(
            self.tree_histograms["node_rank"],
            get_path(tree, "probability.node_rank_histogram"),
        )
        self.accepted_ended_at_leaf[str(bool(tree.get("accepted_ended_at_leaf")))] += 1

    def _record_leaf_redraft(self, leaf_redraft: dict[str, Any]) -> None:
        self.leaf["leaf_count"].add(leaf_redraft.get("leaf_count"))
        add_histogram(
            self.leaf_histograms["path_length"],
            leaf_redraft.get("path_length_histogram"),
        )

        groups = leaf_redraft.get("groups") or []
        self.leaf["groups_per_round"].add(len(groups))
        for group in groups:
            self.total_leaf_groups += 1
            self.leaf["group.batch_size"].add(group.get("batch_size"))
            self.leaf["group.path_length"].add(group.get("path_length"))
            self.leaf["group.context_length"].add(group.get("context_length"))
            for row in group.get("sampled_token_ids", []):
                if row:
                    self.leaf_sampled_first_tokens[str(row[0])] += 1
            self._record_tensor_stats(
                "leaf_redraft.noise_embedding",
                group.get("noise_embedding"),
            )
            self._record_tensor_stats("leaf_redraft.draft_hidden", group.get("draft_hidden"))
            self._record_hidden_stack(
                "leaf_redraft.draft_hidden_states",
                group.get("draft_hidden_states"),
            )
            self._record_tensor_stats("leaf_redraft.draft_logits", group.get("draft_logits"))

    def _record_summary_fields(self, prefix: str, summary: Optional[dict[str, Any]]) -> None:
        if not isinstance(summary, dict):
            return
        for field in ["min", "mean", "std", "p50", "p75", "p95", "max"]:
            self.tree[f"{prefix}.{field}"].add(summary.get(field))

    def _record_tensor_stats(self, prefix: str, stats: Optional[dict[str, Any]]) -> None:
        if not isinstance(stats, dict):
            return

        for field in [
            "finite_fraction",
            "mean",
            "std",
            "rms",
            "l1_mean",
            "abs_max",
            "zero_fraction",
            "abs_lt_1e_6_fraction",
            "abs_lt_1e_4_fraction",
            "abs_lt_1e_2_fraction",
        ]:
            self.tensor[f"{prefix}.{field}"].add(stats.get(field))

        quantiles = stats.get("quantiles") or {}
        for key in ["p00", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p100"]:
            self.tensor[f"{prefix}.quantile.{key}"].add(quantiles.get(key))

        distribution = stats.get("distribution") or {}
        for name, nested in distribution.items():
            self._record_tensor_stats(f"{prefix}.distribution.{name}", nested)

    def _record_hidden_stack(self, prefix: str, stack: Optional[list[dict[str, Any]]]) -> None:
        if not isinstance(stack, list):
            return
        for item in stack:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            selected = "selected" if item.get("selected_for_dflash") else "all"
            layer_prefix = f"{prefix}.layer_{index}.{selected}"
            self._record_tensor_stats(layer_prefix, item.get("stats"))

    def _record_interesting_round(
        self,
        *,
        benchmark_name: str,
        prompt_index: int,
        turn_index: int,
        tree: dict[str, Any],
    ) -> None:
        row = {
            "benchmark": benchmark_name,
            "prompt": prompt_index,
            "turn": turn_index,
            "round": tree.get("round_index"),
            "decode_start": tree.get("decode_start_index"),
            "accepted_len": tree.get("accepted_path_length_with_root"),
            "accepted_depth": tree.get("accepted_path_depth"),
            "ended_at_leaf": tree.get("accepted_ended_at_leaf"),
            "max_depth": tree.get("max_depth"),
            "max_width": tree.get("max_width"),
            "leaf_count": tree.get("leaf_count"),
            "nodes": tree.get("node_count_with_root"),
        }
        self.interesting_rounds.append(row)

    def compact_summary(self) -> dict[str, Any]:
        return {
            "benchmarks": self.benchmarks,
            "counts": {
                "prompts": self.total_prompts,
                "turns": self.total_turns,
                "rounds": self.total_rounds,
                "leaf_redraft_groups": self.total_leaf_groups,
            },
            "acceptance": {
                key: series.summary() for key, series in sorted(self.acceptance.items())
            },
            "stage_times": {
                key: series.summary() for key, series in sorted(self.stage_times.items())
            },
            "tree": {key: series.summary() for key, series in sorted(self.tree.items())},
            "tree_histograms": {
                key: dict(counter)
                for key, counter in sorted(self.tree_histograms.items())
            },
            "accepted_ended_at_leaf": dict(self.accepted_ended_at_leaf),
            "leaf_redraft": {
                key: series.summary() for key, series in sorted(self.leaf.items())
            },
            "leaf_histograms": {
                key: dict(counter)
                for key, counter in sorted(self.leaf_histograms.items())
            },
            "leaf_sampled_first_tokens": dict(self.leaf_sampled_first_tokens),
            "tensor": {
                key: series.summary() for key, series in sorted(self.tensor.items())
            },
            "interesting_rounds": self._top_interesting_rounds(),
        }

    def _top_interesting_rounds(self) -> dict[str, list[dict[str, Any]]]:
        rows = self.interesting_rounds
        limit = self.top_rounds

        def numeric(row, key, default=0):
            value = row.get(key)
            return default if value is None else value

        return {
            "lowest_acceptance": sorted(
                rows,
                key=lambda row: (
                    numeric(row, "accepted_len", 10**9),
                    -numeric(row, "max_depth"),
                    -numeric(row, "leaf_count"),
                ),
            )[:limit],
            "deepest_tree": sorted(
                rows,
                key=lambda row: (
                    -numeric(row, "max_depth"),
                    -numeric(row, "leaf_count"),
                    numeric(row, "accepted_len", 10**9),
                ),
            )[:limit],
            "widest_tree": sorted(
                rows,
                key=lambda row: (
                    -numeric(row, "max_width"),
                    -numeric(row, "leaf_count"),
                    numeric(row, "accepted_len", 10**9),
                ),
            )[:limit],
        }

    def markdown_report(self, *, metadata: dict[str, Any], source_path: Path) -> str:
        summary = self.compact_summary()
        lines = [
            "# DFlash/DDTree Experimental Eval Summary",
            "",
            "## Run",
            "",
            f"- Source: `{source_path}`",
            f"- exp_tag: `{metadata.get('exp_tag', '-')}`",
            f"- options: {brief_options(metadata.get('exp_options'))}",
            f"- target: `{metadata.get('target_model_path', '-')}`",
            f"- draft: `{metadata.get('draft_model_path', '-')}`",
            f"- draft_type: `{metadata.get('draft_type', '-')}`",
            f"- apply_ddtree: `{metadata.get('apply_ddtree', '-')}`",
            f"- ddtree_size: `{metadata.get('ddtree_size', '-')}`",
            "",
            "## Benchmarks",
            "",
            self._benchmark_table(),
            "",
            "## Counts",
            "",
            table(
                ["prompts", "turns", "rounds", "leaf redraft groups"],
                [
                    [
                        summary["counts"]["prompts"],
                        summary["counts"]["turns"],
                        summary["counts"]["rounds"],
                        summary["counts"]["leaf_redraft_groups"],
                    ]
                ],
            ),
            "",
            "## Acceptance",
            "",
            self._series_table(
                [
                    ("prompt accept length", self.acceptance["prompt.accept_length"]),
                    ("turn accept length", self.acceptance["turn.accept_length"]),
                    ("round acceptance length", self.acceptance["round.acceptance_length"]),
                    ("turn new tokens", self.acceptance["turn.num_new_tokens"]),
                    (
                        "turn speculation steps",
                        self.acceptance["turn.num_speculation_steps"],
                    ),
                ]
            ),
            "",
            "## Stage Times",
            "",
            self._stage_time_table(),
            "",
            "## Option 1: DDTree Structure",
            "",
            self._tree_section(),
            "",
            "## Option 2: Leaf Redraft",
            "",
            self._leaf_section(),
            "",
            "## Option 3: Hidden/Logit Tensor Stats",
            "",
            self._tensor_section(),
            "",
            "## Notable Rounds",
            "",
            self._interesting_rounds_section(summary["interesting_rounds"]),
            "",
            "## Reading Guide",
            "",
            "- `accepted_path_length_with_root` includes the root token already verified by the target.",
            "- `accepted_draft_token_count` is accepted path length minus the root.",
            "- Tree probability fields come from DFlash draft logits used to build the tree.",
            "- Tensor rows aggregate the summaries that were already written by the exp eval; they are not raw activations.",
        ]
        return "\n".join(lines).rstrip() + "\n"

    def _benchmark_table(self) -> str:
        rows = []
        for benchmark in self.benchmarks:
            rows.append(
                [
                    benchmark.get("benchmark"),
                    benchmark.get("num_samples"),
                    benchmark.get("generation_mode"),
                    benchmark.get("ddtree_size"),
                    fmt_num(benchmark.get("accept_length")),
                    fmt_num(benchmark.get("output_throughput")),
                    fmt_num(benchmark.get("latency")),
                    fmt_num(benchmark.get("accuracy")),
                    ",".join(str(item) for item in benchmark.get("exp_options") or []),
                ]
            )
        return table(
            [
                "benchmark",
                "samples",
                "mode",
                "tree budget",
                "accept len",
                "tok/s",
                "latency",
                "accuracy",
                "options",
            ],
            rows,
        )

    def _series_table(self, items: list[tuple[str, NumberSeries]]) -> str:
        rows = [stat_row(name, series.summary()) for name, series in items]
        return table(["metric", "count", "mean", "p50", "p95", "max"], rows)

    def _stage_time_table(self) -> str:
        rows = []
        total_mean = sum(series.summary().get("mean", 0.0) for series in self.stage_times.values())
        for name, series in sorted(self.stage_times.items()):
            summary = series.summary()
            mean = summary.get("mean")
            share = mean / total_mean if total_mean and mean is not None else None
            rows.append(
                [
                    name,
                    summary.get("count", 0),
                    fmt_num(mean, 6),
                    fmt_num(summary.get("p50"), 6),
                    fmt_num(summary.get("p95"), 6),
                    fmt_percent(share),
                ]
            )
        return table(["stage", "count", "mean s", "p50 s", "p95 s", "mean share"], rows)

    def _tree_section(self) -> str:
        if not self.tree:
            return "_No tree stats found. Was `--option 1` enabled?_"
        scalar_rows = []
        for name in [
            "node_count_with_root",
            "max_depth",
            "max_width",
            "leaf_count",
            "leaf_fraction",
            "accepted_path_length_with_root",
            "accepted_draft_token_count",
            "accepted_path_depth",
            "visibility_density",
            "tree.probability.node_rank.mean",
            "tree.probability.leaf_cumulative_prob.mean",
        ]:
            series = self.tree.get(name)
            if series:
                scalar_rows.append(stat_row(name, series.summary()))

        ended_total = sum(self.accepted_ended_at_leaf.values())
        ended_true = self.accepted_ended_at_leaf.get("True", 0)
        hist_rows = [
            ["depth", format_histogram(self.tree_histograms["depth"], top_k=self.top_hist)],
            [
                "leaf depth",
                format_histogram(self.tree_histograms["leaf_depth"], top_k=self.top_hist),
            ],
            [
                "branching factor",
                format_histogram(
                    self.tree_histograms["branching_factor"],
                    top_k=self.top_hist,
                ),
            ],
            [
                "width by depth",
                format_histogram(
                    self.tree_histograms["width_by_depth_with_root"],
                    top_k=self.top_hist,
                ),
            ],
            [
                "node rank",
                format_histogram(self.tree_histograms["node_rank"], top_k=self.top_hist),
            ],
        ]

        return "\n\n".join(
            [
                table(
                    ["metric", "count", "mean", "p50", "p95", "max"],
                    scalar_rows,
                ),
                f"Accepted ended at leaf: {ended_true}/{ended_total} ({fmt_percent(ended_true / ended_total if ended_total else None)})",
                table(["histogram", f"top {self.top_hist} bins"], hist_rows),
            ]
        )

    def _leaf_section(self) -> str:
        if not self.leaf:
            return "_No leaf-redraft stats found. Was `--option 2` enabled?_"

        rows = []
        for name in [
            "leaf_count",
            "groups_per_round",
            "group.batch_size",
            "group.path_length",
            "group.context_length",
        ]:
            series = self.leaf.get(name)
            if series:
                rows.append(stat_row(name, series.summary()))

        hist_rows = [
            [
                "path length",
                format_histogram(self.leaf_histograms["path_length"], top_k=self.top_hist),
            ],
            [
                "sampled first token id",
                format_histogram(self.leaf_sampled_first_tokens, top_k=self.top_hist),
            ],
        ]
        return "\n\n".join(
            [
                table(["metric", "count", "mean", "p50", "p95", "max"], rows),
                table(["histogram", f"top {self.top_hist} bins"], hist_rows),
            ]
        )

    def _tensor_section(self) -> str:
        if not self.tensor:
            return "_No tensor stats found. Was `--option 3` enabled?_"

        preferred_keys = [
            "prefill.target_logits.distribution.entropy.mean",
            "prefill.target_logits.distribution.top1_prob.mean",
            "prefill.target_logits.distribution.top1_margin.mean",
            "prefill.dflash_context_feature.rms",
            "round.draft.draft_logits.distribution.entropy.mean",
            "round.draft.draft_logits.distribution.top1_prob.mean",
            "round.draft.draft_logits.distribution.top1_margin.mean",
            "round.draft.draft_hidden.rms",
            "round.verify.target_logits.distribution.entropy.mean",
            "round.verify.target_logits.distribution.top1_prob.mean",
            "round.verify.tree_target_hidden.rms",
            "round.commit.next_context_feature.rms",
            "leaf_redraft.draft_logits.distribution.entropy.mean",
            "leaf_redraft.draft_logits.distribution.top1_prob.mean",
            "leaf_redraft.draft_hidden.rms",
        ]
        rows = []
        for key in preferred_keys:
            series = self.tensor.get(key)
            if series:
                rows.append(stat_row(key, series.summary()))

        if not rows:
            for key, series in sorted(self.tensor.items())[:30]:
                rows.append(stat_row(key, series.summary()))

        return table(["metric", "count", "mean", "p50", "p95", "max"], rows)

    def _interesting_rounds_section(
        self,
        interesting_rounds: dict[str, list[dict[str, Any]]],
    ) -> str:
        blocks = []
        for title, rows in interesting_rounds.items():
            table_rows = []
            for row in rows:
                table_rows.append(
                    [
                        row.get("benchmark"),
                        row.get("prompt"),
                        row.get("turn"),
                        row.get("round"),
                        row.get("decode_start"),
                        row.get("accepted_len"),
                        row.get("accepted_depth"),
                        row.get("ended_at_leaf"),
                        row.get("max_depth"),
                        row.get("max_width"),
                        row.get("leaf_count"),
                        row.get("nodes"),
                    ]
                )
            blocks.append(
                "### "
                + title.replace("_", " ").title()
                + "\n\n"
                + table(
                    [
                        "bench",
                        "prompt",
                        "turn",
                        "round",
                        "start",
                        "acc len",
                        "acc depth",
                        "leaf?",
                        "max depth",
                        "max width",
                        "leaves",
                        "nodes",
                    ],
                    table_rows,
                )
            )
        return "\n\n".join(blocks)


def load_full(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    metadata = {key: value for key, value in data.items() if key != "benchmarks"}
    return metadata, data.get("benchmarks", [])


def stream_metadata(path: Path) -> dict[str, Any]:
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("`--stream` requires `ijson` to be installed.") from exc

    metadata: dict[str, Any] = {}
    current_array_key = None
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "benchmarks" and event == "start_array":
                break
            if event in {"string", "number", "boolean", "null"} and "." not in prefix:
                metadata[prefix] = value
            elif event == "start_array" and "." not in prefix:
                current_array_key = prefix
                metadata[current_array_key] = []
            elif event in {"number", "string", "boolean", "null"} and current_array_key:
                if prefix == f"{current_array_key}.item":
                    metadata[current_array_key].append(value)
            elif event == "end_array" and prefix == current_array_key:
                current_array_key = None
    return metadata


def iter_stream_benchmarks(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("`--stream` requires `ijson` to be installed.") from exc

    with path.open("rb") as handle:
        yield from ijson.items(handle, "benchmarks.item")


def analyze(path: Path, *, stream: bool, top_rounds: int, top_hist: int):
    analyzer = ExpAnalyzer(top_rounds=top_rounds, top_hist=top_hist)
    if stream:
        metadata = stream_metadata(path)
        for benchmark in iter_stream_benchmarks(path):
            analyzer.analyze_benchmark(benchmark)
        return metadata, analyzer

    metadata, benchmarks = load_full(path)
    for benchmark in benchmarks:
        analyzer.analyze_benchmark(benchmark)
    return metadata, analyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize large DFlash/DDTree experimental eval JSON files."
    )
    parser.add_argument("result_json", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write a Markdown report to this path. Defaults to stdout.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write a compact machine-readable summary JSON.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream benchmarks with ijson instead of json.load. Requires ijson.",
    )
    parser.add_argument("--top-rounds", type=int, default=8)
    parser.add_argument("--top-hist", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, analyzer = analyze(
        args.result_json,
        stream=args.stream,
        top_rounds=args.top_rounds,
        top_hist=args.top_hist,
    )
    report = analyzer.markdown_report(metadata=metadata, source_path=args.result_json)

    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote Markdown report to {args.output}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "summary": analyzer.compact_summary(),
        }
        args.summary_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote compact JSON summary to {args.summary_json}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
