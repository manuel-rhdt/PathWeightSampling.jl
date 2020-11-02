# test if the analytically computed stationary distribution stays invariant under the
# stochastic simulation

using Catalyst
using Distributions: Normal, pdf, MvNormal
using StatsBase
using Test

corr_time_s = 100
mean_s = 1000
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
signal_stationary = MvNormal([mean_s], sigma_squared_ss .* Matrix{Float64}(I, 1, 1))

sn = @reaction_network begin
    κ, ∅ --> S
    λ, S --> ∅
end κ λ

rn = @reaction_network begin
    ρ, S --> X + S
    μ, X --> ∅ 
end ρ μ

gen = Trajectories.configuration_generator(sn, rn, [κ, λ], [ρ, μ], signal_stationary, joint_stationary)





signal_stationary = Normal(mean_s, sqrt(sigma_squared_ss))
response_stationary = Normal(mean_x, sqrt(sigma_squared_xx))

joint_samples = round.(rand(joint_stationary, 1_000_000))

using HypothesisTests

num_samples = 100

begin_states = map(1:num_samples) do _
    (system, initial) = Trajectories.generate_configuration(gen, duration=1.0)
    Vector(vcat(initial.u[begin], system.response.u[begin]))
end

end_states = map(1:num_samples) do _
    (system, initial) = Trajectories.generate_configuration(gen, duration=500.0)
    Vector(vcat(initial.u[end], system.response.u[end]))
end

begin_states = reduce((x,y)->hcat(x,y), begin_states)
end_states = reduce((x,y)->hcat(x,y), end_states)

test1 = ApproximateTwoSampleKSTest(joint_samples[1,:], begin_states[1,:])
@test pvalue(test1) > 0.01

test2 = ApproximateTwoSampleKSTest(joint_samples[1,:], end_states[1,:])
@test pvalue(test2) > 0.01

test3 = ApproximateTwoSampleKSTest(joint_samples[2,:], begin_states[2,:])
@test pvalue(test3) > 0.01

test4 = ApproximateTwoSampleKSTest(joint_samples[2,:], end_states[2,:])
@test pvalue(test4) > 0.01
