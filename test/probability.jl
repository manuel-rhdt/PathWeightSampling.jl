using Catalyst
using GaussianMcmc.Trajectories
using Test
using StaticArrays

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

dist = Trajectories.distribution(sn)

for i in 1:100
    @test dist.totalrate([i], []) ≈ 0.005 * i + 0.25
end

Trajectories.build_reaction_rate(sn, reactions(sn)...)

traj = Trajectory(SA[:S], [0.0, 1.0, 2.0, 3.0], [SA[50], SA[51], SA[50], SA[50]])

wait1 = - 1.0 * (0.25 + 50 * 0.005) 
wait2 = - 1.0 * (0.25 + 51 * 0.005)
wait3 = - 1.0 * (0.25 + 50 * 0.005)
reac1 = log(0.25)
reac2 = log(51 * 0.005)

@test Trajectories.logpdf(dist, traj) ≈ sum((wait1, wait2, wait3, reac1, reac2))
