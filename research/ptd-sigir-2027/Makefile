PYTHON ?= python3
PYARROW_RUN ?= uv run --with pyarrow python
TECTONIC ?= tectonic
TECTONIC_BUNDLE ?= https://data1.fullyjustified.net/tlextras-2022.0r0.tar
SOURCE_DATE_EPOCH ?= 1790262000

.PHONY: all generate test smoke audit-source finalize validate check-evidence paper paper-review paper-author clean

all: generate test smoke paper finalize validate

generate:
	$(PYTHON) scripts/generate_tables.py
	$(PYTHON) scripts/generate_figures.py

test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/check_citations.py

smoke:
	$(PYTHON) scripts/run_reference_smoke.py

audit-source:
	$(PYARROW_RUN) scripts/inventory_raw_inputs.py
	$(PYARROW_RUN) scripts/audit_item_universe.py

finalize:
	$(PYTHON) scripts/finalize_manifest.py

validate: generate
	$(PYTHON) scripts/validate_artifacts.py

check-evidence:
	@test -n "$(RUN_MANIFEST)" || (echo "RUN_MANIFEST is required" >&2; exit 2)
	@test -n "$(EVALUATION)" || (echo "EVALUATION is required" >&2; exit 2)
	@test -n "$(PAIRED_OBSERVATIONS)" || (echo "PAIRED_OBSERVATIONS is required" >&2; exit 2)
	$(PYTHON) scripts/check_evidence_candidate.py --run-manifest "$(RUN_MANIFEST)" --evaluation "$(EVALUATION)" --paired-observations "$(PAIRED_OBSERVATIONS)"

paper: paper-review paper-author

paper-review: generate
	cd paper && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(TECTONIC) --bundle $(TECTONIC_BUNDLE) --keep-logs --keep-intermediates main.tex
	$(PYTHON) scripts/normalize_pdf.py paper/main.pdf --source-date-epoch $(SOURCE_DATE_EPOCH)

paper-author: generate
	cd paper && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(TECTONIC) --bundle $(TECTONIC_BUNDLE) --keep-logs --keep-intermediates main-author.tex
	$(PYTHON) scripts/normalize_pdf.py paper/main-author.pdf --source-date-epoch $(SOURCE_DATE_EPOCH)

clean:
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out paper/main.pdf
	rm -f paper/main-author.aux paper/main-author.bbl paper/main-author.blg paper/main-author.log paper/main-author.out paper/main-author.pdf
