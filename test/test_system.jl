using GaussianMcmc
using Catalyst
import Statistics: var
import LinearAlgebra: I
import Distributions: MvNormal, Poisson

sn = @reaction_network begin
    κ, ∅ --> S
    λ, S --> ∅
end κ λ

rn = @reaction_network begin
    ρ, S --> X + S
    μ, X --> ∅ 
end ρ μ

κ = 10.0
λ = 1.0
ρ = 5.0
μ = 5.0
mean_s = κ / λ
mean_x = mean_s * ρ / μ

system = GaussianMcmc.JumpSystem(sn, rn, [κ, λ], [ρ, μ], s_mean=mean_s, x_mean=mean_x, duration=1.0)