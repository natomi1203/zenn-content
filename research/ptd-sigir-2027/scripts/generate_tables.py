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
    counts = {status: sum(row["status"] == status for row in rows) for status in ("VERIFIED", "PREREGISTERED", "PENDING")}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Claim status at manuscript generation time.}",
        r"\label{tab:claim-status}",
        r"\centering",
        r"\begin{tabular}{lr}",
        r"\toprule Status & Claims \\ \midrule",
        *(f"{status.title()} & {count} \\\\" for status, count in counts.items()),
        r"\bottomrule\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "claim_status_table.tex").write_text("\n".join(lines) + "\n")


def pending_results() -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Preregistered prospective comparisons. Values remain pending until the evidence gate passes.}",
        r"\label{tab:ptd-results}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule Variant & NDCG@50 & $\Delta$ [95\% CI] & Status \\ \midrule",
        r"Fixed-tree TDM & -- & -- & pending \\",
        r"PTD item sibling & -- & -- & pending \\",
        r"PTD node sibling & -- & -- & pending \\",
        r"PTD item+node & -- & -- & pending \\",
        r"Alternating PTD & -- & -- & pending \\",
        r"\bottomrule\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "pending_results_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    related_work()
    evidence_table()
    status_table()
    pending_results()
