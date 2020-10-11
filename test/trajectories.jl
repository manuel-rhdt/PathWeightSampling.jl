using Test
using GaussianMcmc.Trajectories
using DiffEqJump
using Catalyst
using StaticArrays

traj = Trajectory(SA[:S, :X], [0.0, 1.0, 2.0], [SA[1,4], SA[2,5], SA[3,6]])

@test length(traj) == 3 == length(traj.t)
@test collect(traj) == [([1, 4], 0.0), ([2,5], 1.0), ([3, 6], 2.0)]

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

u0 = SA[50.0]
tspan = (0., 100.)
discrete_prob = DiscreteProblem(u0, tspan)
jump_prob = JumpProblem(sn, discrete_prob, Direct())
sol = solve(jump_prob, SSAStepper())

traj_sol = convert(Trajectory, Trajectories.trajectory(sol, SA[:S], SA[1]))

for i in eachindex(sol)
    @test sol.t[i] == traj_sol.t[i]
    @test sol[i] == traj_sol.u[i]
end

traj2 = Trajectory(SA[:Y, :Z], [0.5, 1.5, 2.5], [SA[1,4], SA[2,5], SA[3,6]])

joint = merge(traj, traj2)

@test collect(joint) == [
    ([1,4,1,4], 0.0),
    ([2,5,1,4], 1.0),
    ([2,5,2,5], 1.5),
    ([3,6,2,5], 2.0),
    ([3,6,3,6], 2.5)
]

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

network = merge(sn, rn)

u0 = SA[50.0,50.0]
tspan = (0., 100.)
discrete_prob = DiscreteProblem(u0, tspan)
jump_prob = JumpProblem(network, discrete_prob, Direct())
sol = solve(jump_prob, SSAStepper())

partial = Trajectories.trajectory(sol, SA[:S], SA[1])
for i in eachindex(sol)
    @test sol.t[i] == partial.t[i]
    @test sol[[1],i] == partial[i]
end
ptraj = convert(Trajectory, partial)
