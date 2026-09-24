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

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ptd_architecture.tex").write_text(ARCH)
    (OUT / "evidence_flow.tex").write_text(FLOW)
