.PHONY: help smoke full clean duck_full duck_reduced bf kernel_compare probit probit_smoke

help:
	@echo "Targets:"
	@echo "  make smoke        - fast end-to-end smoke test (simulation + duck smoke)"
	@echo "  make full         - heavier example run (simulation subset)"
	@echo "  make duck_full    - run full duck model (data analysis)"
	@echo "  make duck_reduced - run reduced duck model (data analysis)"
	@echo "  make clean        - remove results/"
	@echo "  make kernel_compare - run u kernel comparison experiment (simulation)"
	@echo "  make probit_smoke	 - fast smoke test for the probit simulation"
	@echo "  make probit 	     - run full simulation for the fitted value curve"

smoke:
	bash scripts/run_smoke.sh

full:
	bash scripts/run_full.sh

duck_full:
	bash scripts/run_duck_full.sh

duck_reduced:
	bash scripts/run_duck_reduced.sh

clean:
	rm -rf results

kernel_compare:
	bash scripts/run_kernel_compare.sh

probit:
	bash scripts/run_probit.sh

probit_smoke:
	bash scripts/run_probit_smoke.sh