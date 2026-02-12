.PHONY: help smoke full clean duck_full duck_reduced bf

help:
	@echo "Targets:"
	@echo "  make smoke        - fast end-to-end smoke test (simulation + duck smoke)"
	@echo "  make full         - heavier example run (simulation subset)"
	@echo "  make duck_full    - run full duck model (data analysis)"
	@echo "  make duck_reduced - run reduced duck model (data analysis)"
	@echo "  make bf           - run full + reduced + Bayes factor pipeline"
	@echo "  make clean        - remove results/"

smoke:
	bash scripts/run_smoke.sh

full:
	bash scripts/run_full.sh

duck_full:
	bash scripts/run_duck_full.sh

duck_reduced:
	bash scripts/run_duck_reduced.sh

bf:
	bash scripts/run_bayes_factor.sh

clean:
	rm -rf results
