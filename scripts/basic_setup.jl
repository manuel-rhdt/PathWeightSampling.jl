using GaussianMcmc.Trajectories
using Catalyst
using DrWatson

# sn = @reaction_network begin
#     0.005, S --> ∅
#     0.25, ∅ --> S
# end

# rn = @reaction_network begin
#     0.01, S --> X + S
#     0.01, X --> ∅ 
# end

sn = @reaction_network begin
    κ, ∅ --> S
    λ, s --> ∅
end κ λ

rn = @reaction_network begin
    ρ, S --> X + S
    μ, X --> ∅ 
end ρ μ

gen = Trajectories.configuration_generator(sn, rn, [0.005, 0.25], [0.01, 0.01])
(system, initial) = Trajectories.generate_configuration(gen; duration=500.0)

nothing
