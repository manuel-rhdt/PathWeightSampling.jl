using StatsFuns
using Test

include("test_system.jl")

(system, initial) = Trajectories.generate_configuration(gen, duration=300.0)

annealing = AnnealingEstimate(10, 50, 1000)
ti = TIEstimate(1024, 16, 2^14)

value1 = Trajectories.simulate(annealing, initial, system)
value2 = Trajectories.simulate(ti, initial, system)
@test isapprox(log_marginal(value1), log_marginal(value2); atol=5*sqrt(var(value1) + var(value2)))
@test sqrt(var(value1)) <= abs(1e-4 * log_marginal(value1))
@test sqrt(var(value2)) <= abs(1e-4 * log_marginal(value2))
