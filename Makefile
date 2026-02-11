.PHONY: help smoke full clean

help:
	@echo "Targets:"
	@echo "  make smoke   - run a fast end-to-end smoke test"
	@echo "  make full    - run a heavier example (not full paper grid)"
	@echo "  make clean   - remove results/"

smoke:
	bash scripts/run_smoke.sh

full:
	bash scripts/run_full.sh

clean:
	rm -rf results
