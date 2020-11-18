using Catalyst
using GaussianMcmc.Trajectories
using Test
using StaticArrays
using Distributions

sn = @reaction_network begin
    κ, ∅ --> S
    λ, S --> ∅
end κ λ

s0_dist = MvNormal([50.0], 50.0 * ones((1, 1)))
log_p0 = (s) -> logpdf(s0_dist, [s])

dist = Trajectories.distribution(sn, log_p0)

traj = Trajectory(SA[:S], [0.0, 1.0, 2.0, 3.0], [SA[50.0], SA[51.0], SA[50.0], SA[50.0]])

κ = 0.25
λ = 0.005

p_wait(s, dt) = exp(- (κ + s * λ) * dt)

p0    = log_p0(50.0)
wait1 = log(p_wait(50, 1.0))
wait2 = log(p_wait(51, 1.0))
wait3 = log(p_wait(50, 1.0))
reac1 = log(κ)
reac2 = log(51 * λ)

@test Trajectories.logpdf(dist, traj, params=[κ, λ]) ≈ sum((p0, wait1, wait2, wait3, reac1, reac2))
