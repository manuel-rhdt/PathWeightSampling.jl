using Catalyst
using GaussianMcmc.Trajectories
using Test
using StaticArrays
using Distributions

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

s0_dist = MvNormal([50.0], [50.0])
log_p0 = (s) -> logpdf(s0_dist, [s])

dist = Trajectories.distribution(sn, log_p0)

traj = Trajectory(SA[:S], [0.0, 1.0, 2.0, 3.0], [SA[50.0], SA[51.0], SA[50.0], SA[50.0]])

p0    = log_p0(50.0)
wait1 = - 1.0 * (0.25 + 50 * 0.005) 
wait2 = - 1.0 * (0.25 + 51 * 0.005)
wait3 = - 1.0 * (0.25 + 50 * 0.005)
reac1 = log(0.25)
reac2 = log(51 * 0.005)

@test Trajectories.logpdf(dist, traj) ≈ sum((p0, wait1, wait2, wait3, reac1, reac2))
