
using Catalyst
using Distributions: Normal, logpdf, pdf, MvNormal
using StatsBase
using GaussianMcmc.Trajectories
using StaticArrays

corr_time_s = 100
mean_s = 100
corr_time_ratio = 5

λ = 1 / corr_time_s
κ = mean_s * λ
μ = corr_time_ratio / corr_time_s
ρ = μ
mean_x = mean_s

# see Tostevin, ten Wolde, eq. 27
sigma_squared_ss = mean_s
sigma_squared_sx = ρ * mean_s / (λ + μ)
sigma_squared_xx = mean_x * (1 + ρ / (λ + μ))

joint_stationary = MvNormal([mean_s, mean_x], [sigma_squared_ss sigma_squared_sx; sigma_squared_sx sigma_squared_xx])

signal_stationary = Normal(mean_s, sqrt(sigma_squared_ss))
response_stationary = Normal(mean_x, sqrt(sigma_squared_xx))

entropy(signal_stationary) + entropy(response_stationary) - entropy(joint_stationary)

sn = @reaction_network begin
    κ, ∅ --> S
    λ, S --> ∅
end κ λ

rn = @reaction_network begin
    ρ, S --> X + S
    μ, X --> ∅ 
end ρ μ

gen = Trajectories.configuration_generator(sn, rn, [κ, λ], [ρ, μ], mean_s, mean_x)
jump_problem = gen.joint_j_problem

using Plots

grid = vcat.(50.0:150.0, ( 50.0:150.0 )')
val = map(x -> pdf(gen.p0_dist, x), grid)
gr()
heatmap(50.0:150.0, 50.0:150.0, val, c=:bilbao)

