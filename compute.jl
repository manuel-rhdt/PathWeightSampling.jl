using GaussianMcmc.Trajectories
using Catalyst
using StaticArrays
using DifferentialEquations
using Statistics 

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

me = Trajectories.marginal_entropy(sn, rn, 1000, 500, 16)
ce = Trajectories.conditional_entropy(sn, rn, 1_000_000)

