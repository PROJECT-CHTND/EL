from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


SCORES_CSV_PATH = Path("reports/prompt_runs/scores.csv")
OUTPUT_MD_PATH = Path("reports/weekly.md")


def ensure_reports_directory() -> None:
    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)


def parse_bool(value: str | int | None) -> bool:
    if value is None:
        return False
    if isinstance(value, int):
        return value != 0
    v = str(value).strip().lower()
    return v in {"1", "true", "yes", "y", "t"}


@dataclass
class PromptStats:
    total: int = 0
    ok_count: int = 0
    form_ng_count: int = 0
    latencies: List[float] | None = None

    def __post_init__(self) -> None:
        if self.latencies is None:
            self.latencies = []

    def add(self, ok: bool, form_ng: bool, latency: float) -> None:
        self.total += 1
        if ok:
            self.ok_count += 1
        if form_ng:
            self.form_ng_count += 1
        self.latencies.append(latency)

    def ok_rate(self) -> float:
        return (self.ok_count / self.total) if self.total else 0.0

    def form_ng_rate(self) -> float:
        return (self.form_ng_count / self.total) if self.total else 0.0

    def mean_latency(self) -> float:
        return statistics.fmean(self.latencies) if self.latencies else 0.0


def read_scores(csv_path: Path) -> Tuple[Dict[str, PromptStats], int, str | None, str | None]:
    stats_by_prompt: Dict[str, PromptStats] = defaultdict(PromptStats)
    total_rows = 0
    min_stamp: str | None = None
    max_stamp: str | None = None

    if not csv_path.exists():
        return {}, 0, None, None

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            prompt = (row.get("prompt") or "").strip() or "(unknown)"
            # latency may be int or float in string
            try:
                latency = float(row.get("latency") or 0.0)
            except ValueError:
                latency = 0.0

            ok = parse_bool(row.get("ok"))
            form_ng = parse_bool(row.get("form_ng"))

            stats_by_prompt[prompt].add(ok=ok, form_ng=form_ng, latency=latency)

            stamp = row.get("stamp")
            if stamp:
                if min_stamp is None or stamp < min_stamp:
                    min_stamp = stamp
                if max_stamp is None or stamp > max_stamp:
                    max_stamp = stamp

    return stats_by_prompt, total_rows, min_stamp, max_stamp


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_markdown(
    stats_by_prompt: Dict[str, PromptStats],
    total_rows: int,
    min_stamp: str | None,
    max_stamp: str | None,
) -> str:
    header_lines = [
        "# Weekly Prompt Report",
        "",
    ]
    if total_rows == 0:
        header_lines += ["", "No data found in reports/prompt_runs/scores.csv."]
        return "\n".join(header_lines) + "\n"

    if min_stamp and max_stamp:
        header_lines.append(f"Data range: {min_stamp} — {max_stamp}")
    header_lines.append(f"Total samples: {total_rows}")
    header_lines.append("")

    table_lines = [
        "| Prompt | Samples | OK rate | Form NG rate | Mean latency |",
        "|---|---:|---:|---:|---:|",
    ]

    # Sort by OK rate desc, then by prompt name
    for prompt, st in sorted(
        stats_by_prompt.items(), key=lambda kv: (kv[1].ok_rate(), kv[0]), reverse=True
    ):
        table_lines.append(
            "| {prompt} | {n} | {ok} | {form_ng} | {latency:.3f} |".format(
                prompt=prompt.replace("|", "/"),
                n=st.total,
                ok=format_percentage(st.ok_rate()),
                form_ng=format_percentage(st.form_ng_rate()),
                latency=st.mean_latency(),
            )
        )

    return "\n".join(header_lines + table_lines) + "\n"


def write_markdown(md_text: str) -> None:
    ensure_reports_directory()
    OUTPUT_MD_PATH.write_text(md_text, encoding="utf-8")


def main() -> None:
    stats_by_prompt, total_rows, min_stamp, max_stamp = read_scores(SCORES_CSV_PATH)
    md = generate_markdown(stats_by_prompt, total_rows, min_stamp, max_stamp)
    write_markdown(md)


if __name__ == "__main__":
    main()


