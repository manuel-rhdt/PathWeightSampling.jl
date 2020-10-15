using GaussianMcmc.Trajectories
using Catalyst
using DrWatson

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

gen = Trajectories.configuration_generator(sn, rn)

(system, initial) = Trajectories.generate_configuration(gen; duration=500.0)
nothing
