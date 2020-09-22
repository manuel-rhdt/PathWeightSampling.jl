using Test
using GaussianMcmc.Trajectories
using Catalyst
using DifferentialEquations
using StaticArrays

traj = Trajectory(SA[:S, :X], [0.0, 1.0, 2.0], [SA[1,4], SA[2,5], SA[3,6]])

@test length(traj) == 3 == length(traj.t)
@test collect(traj) == [(0.0, [1, 4]), (1.0, [2,5]), (2.0, [3, 6])]

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

u0 = SA[50]
tspan = (0., 100.)
discrete_prob = DiscreteProblem{SVector{1,Int64}}(sn, u0, tspan)
jump_prob = JumpProblem(sn, discrete_prob, Direct())
sol = solve(jump_prob, SSAStepper())

traj_sol = Trajectories.trajectory(sol)

for i in eachindex(sol)
    @test sol.t[i] == traj_sol.t[i]
    @test sol[i] == traj_sol.u[i]
end

collect(traj_sol)

traj2 = Trajectory(SA[:Y, :Z], [0.5, 1.5, 2.5], [SA[1,4], SA[2,5], SA[3,6]])

joint = merge(traj, traj2)

collect(joint)
