using GaussianMcmc.Trajectories
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

κ = 0.25
λ = 0.005
ρ = 0.01
μ = 0.01
mean_s = κ / λ
mean_x = mean_s * ρ / μ

# see Tostevin, ten Wolde, eq. 27
sigma_squared_ss = mean_s
sigma_squared_sx = ρ * mean_s / (λ + μ)
sigma_squared_xx = mean_x * (1 + ρ / (λ + μ))

joint_stationary = MvNormal([mean_s, mean_x], [sigma_squared_ss sigma_squared_sx; sigma_squared_sx sigma_squared_xx])
signal_stationary = Poisson(mean_s)

gen = Trajectories.configuration_generator(sn, rn, [κ, λ], [ρ, μ], signal_stationary, joint_stationary)