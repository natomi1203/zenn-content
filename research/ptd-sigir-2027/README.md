# From Ranking to Retrieval: Purchase-Aware Tree Distillation for Recommendation

This directory is the research workspace for a completed negative/null RetailRocket external-validation paper for SIGIR 2027. It separates verified evidence, historical preregistrations, unsupported hypotheses, out-of-scope comparisons, and future work.

- [English progress and reproduction guide](README_EN.md)
- [Japanese progress and reproduction guide](README_JA.md)
- [Paper source](paper/main.tex)
- [Preregistration](artifact/preregistration.md)
- [Claim--evidence ledger](artifact/claim_evidence_ledger.csv)
- [Artifact manifest](artifact/manifest.json) and [schema](artifact/manifest.schema.json)

The manuscript's empirical scope is fixed to the admitted 24-cell RetailRocket Category C validation over widths 300/600/1200/2400. Kauche PTD evaluation is outside the paper's scope and is not required to complete its conclusions.

The RetailRocket evidence index at `artifact/verified/public_retailrocket_evidence_index.json` preserves all 24 admitted cells, evaluation artifacts, immutable object identities, and the complete non-evidence lineage. The accompanying failure analysis is descriptive and does not alter the confirmatory conclusion.

The next research allocation is PTD 20--30% and KMCR 70--80%. The preregistered KMCR direction tests whether bounded complementary candidate injection can recover relevant items already absent from the base retriever's candidate set; see `artifact/kmcr_research_direction.md`.
