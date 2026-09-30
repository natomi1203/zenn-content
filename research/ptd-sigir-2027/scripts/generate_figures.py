#!/usr/bin/env python3
"""Generate deterministic TikZ figures without external plotting packages."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"

ARCH = r"""\begin{figure*}[t]
\centering
\begin{tikzpicture}[node distance=7mm and 9mm, every node/.style={font=\small}, box/.style={draw,rounded corners,align=center,minimum height=8mm,minimum width=25mm}, arr/.style={-Latex,thick}]
\node[box] (seq) {Point-in-time\\action sequence};
\node[box,right=of seq] (hstu) {HSTU-style\\user encoder};
\node[box,right=of hstu] (tree) {Balanced tree\\node scorer};
\node[box,right=of tree] (beam) {Top-down\\beam retrieval};
\node[box,below=of hstu] (teacher) {Frozen legacy ESMM\\$p_{CTR}\,p_{CVR}$};
\node[box,below=of tree] (siblings) {Item and node\\sibling targets};
\node[box,below=of beam] (alt) {Train-only capacity-\\balanced reassignment};
\draw[arr] (seq) -- (hstu);
\draw[arr] (hstu) -- (tree);
\draw[arr] (tree) -- (beam);
\draw[arr] (teacher) -- (siblings);
\draw[arr] (siblings) -- node[right]{KL} (tree);
\draw[arr,dashed] (tree) -- (alt);
\draw[arr,dashed] (alt) -- (tree);
\node[align=center,below=2mm of alt,font=\footnotesize] {Tree locked before test scoring};
\end{tikzpicture}
\caption{Registered PTD design. Solid arrows are model/data flow; dashed arrows occur only during train-side alternating optimization. The teacher is frozen.}
\label{fig:architecture}
\end{figure*}
"""

FLOW = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}[node distance=4mm, every node/.style={font=\footnotesize}, box/.style={draw,rounded corners,align=center,minimum width=40mm}, arr/.style={-Latex}]
\node[box] (run) {Run outputs};
\node[box,below=of run] (manifest) {Run manifest + artifact hashes};
\node[box,below=of manifest] (eval) {Evaluation JSON + paired rows};
\node[box,below=of eval] (gate) {Hash check + recomputation};
\node[box,below=of gate] (ledger) {Outcome-aware claim ledger};
\node[box,below=of ledger] (paper) {Generated tables and prose};
\draw[arr] (run)--(manifest);\draw[arr] (manifest)--(eval);\draw[arr] (eval)--(gate);\draw[arr] (gate)--(ledger);\draw[arr] (ledger)--(paper);
\end{tikzpicture}
\caption{Evidence admission and interpretation. Invalid or irreproducible bundles stop at the gate; valid negative outcomes enter the ledger without supporting positive claims.}
\label{fig:evidence-flow}
\end{figure}
"""

FAILURE = r"""\begin{figure*}[t]
\centering
\begin{tikzpicture}[node distance=7mm and 8mm, every node/.style={font=\footnotesize}, obs/.style={draw,rounded corners,align=center,minimum width=31mm,minimum height=13mm}, unknown/.style={draw,dashed,rounded corners,align=center,minimum width=31mm,minimum height=13mm}, arr/.style={-Latex,thick}]
\node[obs] (agg) {Aggregation rule\\max vs. true-sum\\\textbf{observed}};
\node[obs,right=of agg] (prune) {Finite-beam pruning\\first depth 11/12/13/14\\\textbf{observed}};
\node[obs,right=of prune] (overlap) {Candidate overlap\\mean Jaccard 0.151--0.201\\\textbf{observed}};
\node[obs,right=of overlap] (recall) {Recall@600 difference\\$-0.003/-0.022/-0.023/-0.041$\\\textbf{observed}};
\draw[arr] (agg)--(prune);
\draw[arr] (prune)--(overlap);
\draw[arr] (overlap)--(recall);
\node[unknown,below=of overlap] (causal) {Recovery of already pruned positives\\causal mediation not identified};
\draw[arr,dashed] (prune)--(causal);
\draw[arr,dashed] (causal)--(recall);
\node[unknown,below=of agg] (kmcr) {KMCR complement injection\\future preregistered direction};
\draw[arr,dashed] (kmcr)--(causal);
\end{tikzpicture}
\caption{Descriptive failure-mechanism boundary. Solid boxes/arrows summarize observed 24-cell artifacts; dashed paths are unverified causal explanations or future work. The low overlap shows that the null/negative result is not candidate-set convergence, while wider search delayed first pruning without producing a true-sum advantage.}
\label{fig:failure-mechanism}
\end{figure*}
"""

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ptd_architecture.tex").write_text(ARCH)
    (OUT / "evidence_flow.tex").write_text(FLOW)
    (OUT / "failure_mechanism.tex").write_text(FAILURE)
