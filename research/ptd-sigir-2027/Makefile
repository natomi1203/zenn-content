PYTHON ?= python3
TECTONIC ?= tectonic
TECTONIC_BUNDLE ?= https://data1.fullyjustified.net/tlextras-2022.0r0.tar

.PHONY: all generate validate paper clean

all: generate validate paper

generate:
	$(PYTHON) scripts/generate_tables.py
	$(PYTHON) scripts/generate_figures.py

validate: generate
	$(PYTHON) scripts/validate_artifacts.py

paper: generate
	cd paper && $(TECTONIC) --bundle $(TECTONIC_BUNDLE) --keep-logs --keep-intermediates main.tex

clean:
	rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out paper/main.pdf
