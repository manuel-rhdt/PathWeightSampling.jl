module GaussianMcmcTest

using GaussianMcmc
using Test

system = System()
t = GaussianMcmc.time_matrix(100, 0.1)
initial = GaussianProposal(rand(200))

function test_marginal_dens()
    samples, acc = estimate_marginal_density(initial, 10, system, t; scale = 0.006, skip = 10, θ = 0.5)

    @test length(samples) == 10
    @test length(acc) == 10
end

function test_distributions()
    # three mathematically identical ways to compute the joint logpdf
    # log(P(s,x)) = log(P(x|s)) + log(P(s)) = log(P(s|x)) + log(P(x))
    response = rand(size(t)...)
    signal = rand(size(t)...)

    val1 = log_joint(system, t, signal=signal, response=response)
    val2 = log_likelihood(system, t, signal=signal, response=response) + log_prior(system, t, signal=signal)

    @test size(val1) == (size(t, 1),)
    @test size(val2) == (size(t, 1),)
    @test val1 ≈ val2
end


test_marginal_dens()
test_distributions()

end

include("trajectories.jl")