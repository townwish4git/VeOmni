# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Analyze Muon benchmark result lines in node0.log using only the standard library."""

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


CASE = r"(?:fsdp|hsdp)_dedup[01]_overlap[01]_(?:legacy|vanilla)"
RESULT = re.compile(
    rf"(?P<case>{CASE}):\s*(?P<mean>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*ms,\s*overlap=(?P<status>[A-Za-z_]+)\b"
)
COMMAND_CASE = re.compile(rf"(?:^|\s)--case(?:=|\s+)(?P<case>{CASE})(?=\s|$)")
COMMAND_TRIAL = re.compile(r"(?:^|\s)--trial(?:=|\s+)(\d+)(?=\s|$)")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _parse(log):
    records, notes, seen = {}, [], defaultdict(int)
    context = None
    with log.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n"):
                notes.append(f"第 {line_number} 行尚无换行，可能正在写入，本次暂不读取。")
                continue
            line = ANSI.sub("", raw)
            if "train_muon_step.py" in line and ("torch.distributed.run" in line or "torchrun" in line):
                case_match, trial_match = COMMAND_CASE.search(line), COMMAND_TRIAL.search(line)
                context = (case_match["case"], int(trial_match[1])) if case_match and trial_match else None
            match = RESULT.search(line)
            if not match:
                continue
            case, mean = match["case"], float(match["mean"])
            if not math.isfinite(mean) or mean <= 0:
                notes.append(f"第 {line_number} 行耗时非有限正数，已忽略。")
                continue
            if context and context[0] == case:
                trial, trial_source = context[1], "torchrun --trial"
            else:
                trial, trial_source = seen[case], "按该配置结果行的出现次数推断"
                notes.append(
                    "部分结果行缺少对应 torchrun 命令；按各配置出现次数推断轮次，截断或拼接日志可能导致轮次不准。"
                )
            seen[case] = max(seen[case], trial + 1)
            scenario, dedup, overlap, momentum = case.split("_")
            key = (trial, case)
            if key in records:
                notes.append(f"r{trial:02d} {case} 出现多条结果，保留最后一条（第 {line_number} 行）。")
            records[key] = {
                "trial": trial,
                "case": case,
                "scenario": scenario,
                "dedup": int(dedup[-1]),
                "overlap": int(overlap[-1]),
                "momentum": momentum,
                "mean_ms": mean,
                "overlap_status": match["status"],
                "speedup_vs_baseline": "",
                "latency_reduction_vs_baseline_pct": "",
                "log_line": line_number,
                "trial_source": trial_source,
            }
    return list(records.values()), list(dict.fromkeys(notes))


def _compare(rows):
    by_case = {row["case"]: row for row in rows}
    effects = []
    for row in rows:
        baseline = by_case.get(f"{row['scenario']}_dedup1_overlap0_legacy")
        if baseline:
            row["speedup_vs_baseline"] = baseline["mean_ms"] / row["mean_ms"]
            row["latency_reduction_vs_baseline_pct"] = 100 * (1 - row["mean_ms"] / baseline["mean_ms"])
        for factor, old, new in (("dedup", 1, 0), ("overlap", 0, 1), ("momentum", "legacy", "vanilla")):
            if row[factor] != old or (factor == "dedup" and row["scenario"] == "fsdp"):
                continue
            changed = {**row, factor: new}
            name = "{scenario}_dedup{dedup}_overlap{overlap}_{momentum}".format(**changed)
            after = by_case.get(name)
            if after:
                effects.append(
                    {
                        "scenario": row["scenario"],
                        "factor": factor,
                        "before": row["case"],
                        "after": name,
                        "before_mean_ms": row["mean_ms"],
                        "after_mean_ms": after["mean_ms"],
                        "speedup": row["mean_ms"] / after["mean_ms"],
                        "latency_reduction_pct": 100 * (1 - after["mean_ms"] / row["mean_ms"]),
                        "before_overlap_status": row["overlap_status"],
                        "after_overlap_status": after["overlap_status"],
                        "before_log_line": row["log_line"],
                        "after_log_line": after["log_line"],
                    }
                )
    return effects


def _table(headers, records):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(str(value) for value in row) + " |" for row in records
    ]


def _number(value, suffix=""):
    return f"{value:.3f}{suffix}" if isinstance(value, (float, int)) else "—"


def _csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Print and save the selected repeat's logged performance and matched ablations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to coordinator/node0.log")
    parser.add_argument("--trial", type=int, default=0, help="Zero-based repeat index; default: first repeat")
    parser.add_argument("--scenario", choices=("all", "fsdp", "hsdp"), default="all")
    parser.add_argument("--cases", nargs="+", help="Only these case names for a selected-case suite")
    parser.add_argument("--output", type=Path, help="Default: <log directory>/node0-analysis-r00")
    args = parser.parse_args()
    if not args.log.is_file() or args.trial < 0:
        parser.error("请指定存在的日志文件，trial 必须非负")
    expected = [
        f"{scenario}_dedup{dedup}_overlap{overlap}_{momentum}"
        for scenario in ("fsdp", "hsdp")
        if args.scenario in ("all", scenario)
        for dedup in (1, 0)
        if scenario == "hsdp" or dedup == 1
        for overlap in (0, 1)
        for momentum in ("legacy", "vanilla")
    ]
    if args.cases:
        if set(args.cases) - set(expected):
            parser.error("--cases 包含无效配置或与 --scenario 不匹配")
        expected = list(dict.fromkeys(args.cases))
    records, notes = _parse(args.log)
    rows = [row for row in records if row["trial"] == args.trial and row["case"] in expected]
    rows.sort(key=lambda row: (row["scenario"], row["mean_ms"]))
    effects = _compare(rows)
    missing = [case for case in expected if case not in {row["case"] for row in rows}]
    lines = [
        f"Muon 日志性能分析：第 {args.trial + 1} 轮（r{args.trial:02d}）",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"日志：{args.log.resolve()}",
        f"已识别本轮配置：{len(rows)}/{len(expected)}。",
        "",
        "仅提取形如 `<case>: 12.345 ms, overlap=executed` 的最终结果行，排除 warmup 日志。",
        "使用日志已打印的平均耗时（保留其打印精度）；日志没有逐 step 样本，不能计算 P95、CV 或显存。",
        "按配置名称配对；日志无法核验模型、拓扑、源码版本等是否相同。单轮结果不能说明跨轮稳定性。",
        "",
    ]
    lines += _table(
        ["配置", "平均 ms", "相对基线", "耗时下降", "overlap 实际状态", "日志行"],
        [
            [
                r["case"],
                _number(r["mean_ms"]),
                _number(r["speedup_vs_baseline"], "x"),
                _number(r["latency_reduction_vs_baseline_pct"], "%"),
                r["overlap_status"],
                r["log_line"],
            ]
            for r in rows
        ],
    )
    for scenario in ("fsdp", "hsdp"):
        available = [row for row in rows if row["scenario"] == scenario]
        if available:
            best = min(available, key=lambda row: row["mean_ms"])
            lines += [
                "",
                f"{scenario.upper()} 当前最低耗时：{best['case']}，{best['mean_ms']:.3f} ms，"
                f"相对基线 {_number(best['speedup_vs_baseline'], 'x')}。",
            ]
    lines += ["", "单因素消融（固定其他开关；dedup 表示 on→off，momentum 表示 legacy→vanilla）：", ""]
    lines += _table(
        ["功能", "before", "after", "加速比", "耗时下降", "after overlap 状态"],
        [
            [
                r["factor"],
                r["before"],
                r["after"],
                _number(r["speedup"], "x"),
                _number(r["latency_reduction_pct"], "%"),
                r["after_overlap_status"],
            ]
            for r in effects
        ],
    )
    for row in rows:
        if row["overlap"] and row["overlap_status"] != "executed":
            notes.append(f"{row['case']}：overlap 状态为 {row['overlap_status']}，开关比较不代表有效 overlap 的收益。")
    if missing:
        lines += ["", "本轮尚未看到结果行的配置（未跑、运行中、失败或日志不完整，不能仅凭此区分）：", ""]
        lines += [f"- {case}" for case in missing]
    if notes:
        lines += ["", "解析及状态提示：", ""] + [f"- {note}" for note in notes]
    if not rows:
        lines += ["", "没有匹配的性能结果。检查日志是否包含上述格式；其他日志格式需提供实际结果行后适配。"]
    output = args.output or args.log.parent / f"{args.log.stem}-analysis-r{args.trial:02d}"
    output.mkdir(parents=True, exist_ok=True)
    _csv(
        output / "summary.csv",
        rows,
        [
            "trial",
            "case",
            "scenario",
            "dedup",
            "overlap",
            "momentum",
            "mean_ms",
            "overlap_status",
            "speedup_vs_baseline",
            "latency_reduction_vs_baseline_pct",
            "log_line",
            "trial_source",
        ],
    )
    _csv(
        output / "ablations.csv",
        effects,
        [
            "scenario",
            "factor",
            "before",
            "after",
            "before_mean_ms",
            "after_mean_ms",
            "speedup",
            "latency_reduction_pct",
            "before_overlap_status",
            "after_overlap_status",
            "before_log_line",
            "after_log_line",
        ],
    )
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"报告保存至：{output.resolve()}")


if __name__ == "__main__":
    main()
