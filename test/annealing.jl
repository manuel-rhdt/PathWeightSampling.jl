using GaussianMcmc.Trajectories
using Catalyst
using StatsFuns
using Test

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

gen = Trajectories.configuration_generator(sn, rn)
(system, initial) = Trajectories.generate_configuration(gen, duration=300.0)

annealing = AnnealingEstimate(10, 50, 1000)
ti = TIEstimate(50, 8, 2000)

(value1, acc) = Trajectories.log_marginal(annealing, initial, system)
(value2, acc) = Trajectories.log_marginal(ti, initial, system)
@test isapprox(value1, value2; rtol=1e-3)
