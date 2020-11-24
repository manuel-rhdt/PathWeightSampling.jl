using StatsFuns
using Test
import GaussianMcmc: log_marginal

include("test_system.jl")

initial = GaussianMcmc.generate_configuration(system)

annealing = AnnealingEstimate(10, 50, 1000)
ti = TIEstimate(1024, 16, 2^14)

value1 = GaussianMcmc.simulate(annealing, initial, system)
value2 = GaussianMcmc.simulate(ti, initial, system)

@test isapprox(log_marginal(value1), log_marginal(value2); atol=5 * sqrt(var(value1) + var(value2)))
@test sqrt(var(value1)) <= abs(1e-4 * log_marginal(value1))
@test sqrt(var(value2)) <= abs(1e-4 * log_marginal(value2))
