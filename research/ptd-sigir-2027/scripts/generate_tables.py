#!/usr/bin/env python3
"""Generate manuscript tables from machine-readable evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"


def esc(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")):
        text = text.replace(old, new)
    return text


def related_work() -> None:
    rows = json.loads((ROOT / "artifact" / "related_work.json").read_text())
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Positioning against primary adjacent work. PTD statements describe the registered design, not empirical superiority.}",
        r"\label{tab:related}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{l X X}",
        r"\toprule",
        r"Work & Prior mechanism & Registered PTD distinction \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['work'])}~\\cite{{{row['citation']}}} & "
            f"{esc(row['retrieval_unit'])}; teacher: {esc(row['teacher'])}; tree: {esc(row['tree_update'])}. & "
            f"{esc(row['relation'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    (OUT / "related_work_table.tex").write_text("\n".join(lines) + "\n")


def evidence_table() -> None:
    evidence = json.loads((ROOT / "artifact" / "verified" / "legacy_esmm_evidence.json").read_text())
    metric = evidence["summary"]["legacy_esmm"]
    newline = r"\\"
    lines = [
        r"\begin{table}[t]",
        r"\caption{Verified legacy context on the common five-date universe. These are not PTD results.}",
        r"\label{tab:legacy}",
        r"\centering",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Verified value \\",
        r"\midrule",
        f"Test rows & {evidence['summary']['test_rows']:,} {newline}",
        f"Purchase-positive pairs & {evidence['summary']['test_positive_pairs']:,} {newline}",
        f"Users with purchase & {evidence['summary']['test_users_with_purchase']:,} {newline}",
        f"Legacy ESMM NDCG@50 & {metric['purchase_ndcg_at_50']:.5f} {newline}",
        f"Legacy ESMM Recall@50 & {metric['purchase_recall_at_50']:.5f} {newline}",
        f"Legacy ESMM AUC & {metric['purchase_auc']:.5f} {newline}",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "legacy_evidence_table.tex").write_text("\n".join(lines) + "\n")


def status_table() -> None:
    with (ROOT / "artifact" / "claim_evidence_ledger.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    statuses = ("VERIFIED", "PREREGISTERED", "UNSUPPORTED", "OUT_OF_SCOPE", "FUTURE_WORK")
    counts = {status: sum(row["status"] == status for row in rows) for status in statuses}
    labels = {
        "VERIFIED": "Verified",
        "PREREGISTERED": "Preregistered",
        "UNSUPPORTED": "Unsupported",
        "OUT_OF_SCOPE": "Out of scope",
        "FUTURE_WORK": "Future work",
    }
    lines = [
        r"\begin{table}[t]",
        r"\caption{Claim status at manuscript generation time.}",
        r"\label{tab:claim-status}",
        r"\centering",
        r"\begin{tabular}{lr}",
        r"\toprule Status & Claims \\ \midrule",
        *(f"{labels[status]} & {count} \\\\" for status, count in counts.items()),
        r"\bottomrule\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "claim_status_table.tex").write_text("\n".join(lines) + "\n")


def public_failure_results() -> None:
    analysis = json.loads(
        (ROOT / "artifact" / "verified" / "public_retailrocket_failure_analysis.json").read_text()
    )
    rows = {row["search_topk"]: row for row in analysis["width_effects"]}
    overlap = {row["search_topk"]: row for row in analysis["candidate_overlap"]}
    pruning = {row["search_topk"]: row for row in analysis["first_prune"]}
    performance = {row["search_topk"]: row for row in analysis["performance"]}
    lines = [
        r"\begin{table*}[t]",
        r"\caption{RetailRocket Category C results (116 purchase users / 65,624 target users; evaluable fraction 0.001768). Intervals by width are descriptive applications of the registered bootstrap; only the width-1200 primary contrast belongs to the registered Holm family ($p_{\mathrm{Holm}}=0.378$).}",
        r"\label{tab:public-failure}",
        r"\centering\scriptsize",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule Width & $\Delta$ Recall@600 & descriptive 95\% CI & mean Jaccard & first prune & p95 ms & peak GiB \\ \midrule",
    ]
    for width in (300, 600, 1200, 2400):
        effect = rows[width]
        ci = effect["descriptive_ci95_registered_bootstrap"]
        lines.append(
            f"{width} & {effect['estimate']:.4f} & [{ci[0]:.4f}, {ci[1]:.4f}] & "
            f"{overlap[width]['mean']:.3f} & {pruning[width]['mean_first_prune_depth']:.0f} & "
            f"{performance[width]['mean_p95_latency_ms']:.0f} & "
            f"{performance[width]['mean_peak_gpu_memory_gib']:.1f} \\\\"
        )
    lines += [r"\bottomrule\end{tabular}", r"\end{table*}"]
    (OUT / "public_failure_results_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    related_work()
    evidence_table()
    status_table()
    public_failure_results()
