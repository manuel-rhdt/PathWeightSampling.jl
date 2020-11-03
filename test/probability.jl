using Catalyst
using GaussianMcmc.Trajectories
using Test
using StaticArrays
using Distributions

sn = @reaction_network begin
    κ, ∅ --> S
    λ, S --> ∅
end κ λ

s0_dist = MvNormal([50.0], 50.0 * ones((1,1)))
log_p0 = (s) -> logpdf(s0_dist, [s])

dist = Trajectories.distribution(sn, log_p0)

traj = Trajectory(SA[:S], [0.0, 1.0, 2.0, 3.0], [SA[50.0], SA[51.0], SA[50.0], SA[50.0]])

κ = 0.25
λ = 0.005

p0    = log_p0(50.0)
wait1 = - 1.0 * (κ + 50 * λ) 
wait2 = - 1.0 * (κ + 51 * λ)
wait3 = - 1.0 * (κ + 50 * λ)
reac1 = log(κ)
reac2 = log(51 * λ)

@test Trajectories.logpdf(dist, traj, params=[κ, λ]) ≈ sum((p0, wait1, wait2, wait3, reac1, reac2))
