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

temps, weights = Trajectories.log_marginal(Val(:Annealing), initial, system, 10, 50, 1000)
value = -(logsumexp(weights[end, :]) - log(size(weights, 2)))
(value2, acc) = Trajectories.log_marginal(initial, system, 2000, 8, 50)
@test isapprox(value, value2; rtol=1e-3)
