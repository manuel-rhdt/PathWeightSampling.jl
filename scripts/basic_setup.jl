using GaussianMcmc
using Catalyst
using DrWatson
using LinearAlgebra

import Distributions: MvNormal, Poisson

sn = @reaction_network begin
    κ, ∅ --> S
    λ, S --> ∅
end κ λ

rn = @reaction_network begin
    ρ, S --> X + S
    μ, X --> ∅ 
end ρ μ

function get_system(mean_s, mean_x, tau_s, tau_x, duration=tau_s)
    κ = mean_s / tau_s
    λ = 1 / tau_s
    ρ = (mean_x / mean_s) / tau_x
    μ = 1 / tau_x
    GaussianMcmc.JumpSystem(sn, rn, [κ, λ], [ρ, μ], mean_s, mean_x, duration)
end

nothing
