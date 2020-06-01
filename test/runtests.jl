module GaussianMcmcTest

using GaussianMcmc
using Test

system = System()
t = GaussianMcmc.time_matrix(100, 0.1)
initial = GaussianProposal(rand(200))

samples, acc = estimate_marginal_density(initial, 10, system, t; scale = 0.006, skip = 10, θ = 0.5)

@test length(samples) == 10
@test length(acc) == 10

end