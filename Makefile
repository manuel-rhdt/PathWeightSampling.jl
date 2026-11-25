# Shortcuts for testing the package and the scripts that reproduce the figures.
# `make help` lists the targets. Set JULIA=/path/to/julia to use another Julia.

JULIA ?= julia
SMOKE ?= $(CURDIR)/.smoke
THREADS ?= auto

.PHONY: help test docs smoke smoke-reproduce smoke-kernels smoke-rates figures clean

help:
	@echo "make test              run the test suite"
	@echo "make docs              build the documentation"
	@echo "make smoke             quick check that all reproduction scripts run (a few minutes)"
	@echo "make smoke-reproduce   figures/figure_rates/reproduce_chemotaxis.jl with tiny settings"
	@echo "make smoke-kernels     figures/figure_kernels/simulate_kernels.jl with tiny settings"
	@echo "make smoke-rates       Gaussian rates from the committed data; must equal the committed values"
	@echo "make figures           draw all figures from the committed data (figures/ environment)"
	@echo "make clean             remove the output of the smoke tests"
	@echo "The full-size reproductions take CPU hours: see figures/figure_rates/README.md and figures/figure_kernels/README.md."

test:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

docs:
	$(JULIA) --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
	$(JULIA) --project=docs docs/make.jl

# --- smoke tests of the scripts: same code paths as the real runs, a tiny number of short trajectories.
# The numbers they produce are not meaningful; they only show that the scripts run to the end.

smoke: smoke-reproduce smoke-kernels smoke-rates

REPRODUCE = $(JULIA) --project=. -t $(THREADS) figures/figure_rates/reproduce_chemotaxis.jl
REPRODUCE_OPTS = --out $(SMOKE)/rates --duration 2 --gradients 2e-4,4e-4 --ntraj 4 --particles 16 --input-trajectories 4

smoke-reproduce:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	$(REPRODUCE) compute $(REPRODUCE_OPTS)
	$(REPRODUCE) assemble $(REPRODUCE_OPTS)
	$(REPRODUCE) input $(REPRODUCE_OPTS)
	$(REPRODUCE) check $(REPRODUCE_OPTS)

KERNELS = $(JULIA) --project=. -t $(THREADS) figures/figure_kernels/simulate_kernels.jl
KERNEL_OPTS = --out $(SMOKE)/kernels --ntraj 4

smoke-kernels:
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	$(KERNELS) noise $(KERNEL_OPTS) --duration 10 --maxlag 200
	$(KERNELS) kernel $(KERNEL_OPTS) --duration 52
	$(KERNELS) assemble $(KERNEL_OPTS)
	$(KERNELS) check $(KERNEL_OPTS)

smoke-rates:
	$(JULIA) --project=figures -e 'using Pkg; Pkg.instantiate()'
	$(JULIA) --project=figures figures/figure_kernels/gaussian_rates.jl --out $(SMOKE)/gaussian --tolerance 1e-9

# --- figures from the committed data

figures:
	$(JULIA) --project=figures -e 'using Pkg; Pkg.instantiate()'
	cd figures && for f in figure_gene_expr/figure_gene_expr.jl figure_kernels/figure_kernels.jl \
	    figure_rates/figure_chemotaxis.jl figure_rates/figure_comparison.jl; do \
	  $(JULIA) --project=. $$f || exit 1; done

clean:
	rm -rf $(SMOKE)
