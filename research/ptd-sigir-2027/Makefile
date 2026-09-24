PYTHON ?= python3
TECTONIC ?= tectonic
TECTONIC_BUNDLE ?= https://data1.fullyjustified.net/tlextras-2022.0r0.tar
SOURCE_DATE_EPOCH ?= 1790262000

.PHONY: all generate test smoke finalize validate paper clean

all: generate test smoke paper finalize validate

generate:
	$(PYTHON) scripts/generate_tables.py
	$(PYTHON) scripts/generate_figures.py

test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/check_citations.py

smoke:
	$(PYTHON) scripts/run_reference_smoke.py

finalize:
	$(PYTHON) scripts/finalize_manifest.py

validate: generate
	$(PYTHON) scripts/validate_artifacts.py

paper: generate
	cd paper && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(TECTONIC) --bundle $(TECTONIC_BUNDLE) --keep-logs --keep-intermediates main.tex
	$(PYTHON) scripts/normalize_pdf.py paper/main.pdf --source-date-epoch $(SOURCE_DATE_EPOCH)

clean:
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out paper/main.pdf
