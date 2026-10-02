# Avirat / SIH26168 IDR - common tasks.   make help
#
# MVP commands use configs/sih_mvp.yaml. `make train` never overwrites the committed MVP
# model (results/models/motionnet.pt): it writes a candidate to $(NEW_MODEL).

PYTHON    ?= python3
CONFIG    ?= configs/sih_mvp.yaml
NEW_MODEL ?= outputs/models/motionnet_candidate.pt
MVP_MODEL := results/models/motionnet.pt

.PHONY: help data baseline train fuse figures replay test all

help:
	@echo "make data      download IO-VNBD (all 72 drives, ~430 MB, checksum-verified)"
	@echo "make baseline  raw INS (A) and filtered INS (B) on the test drives"
	@echo "make train     train MotionNet -> $(NEW_MODEL) (committed MVP model untouched)"
	@echo "make fuse      full system D benchmark (MotionNet + EKF + NHC + map), 30/60/120 s blackouts"
	@echo "make figures   benchmark plus 300-dpi submission figures"
	@echo "make replay    Streamlit replay dashboard"
	@echo "make test      unit tests (no dataset needed)"
	@echo "make all       data, test, baseline, fuse, figures (not train: fuse uses the committed model)"

data:
	$(PYTHON) -m src.download_data

baseline:
	$(PYTHON) -m src.evaluate --config $(CONFIG) --variants A B --tag baseline --plots 0

train:
	@if [ "$(abspath $(NEW_MODEL))" = "$(abspath $(MVP_MODEL))" ]; then \
		echo "refusing to overwrite $(MVP_MODEL); set NEW_MODEL to a new path"; exit 1; fi
	$(PYTHON) -m src.train_motion --config $(CONFIG) --set model.path=$(NEW_MODEL) evaluate.output_dir=outputs/train

fuse:
	$(PYTHON) -m src.evaluate --config $(CONFIG) --tag sih_mvp --plots 0

figures:
	$(PYTHON) -m src.evaluate --config $(CONFIG) --tag sih_mvp --plots 2 --sih-plots

replay:
	streamlit run src/ui/app.py

test:
	$(PYTHON) -m pytest -q

all: data test baseline fuse figures
