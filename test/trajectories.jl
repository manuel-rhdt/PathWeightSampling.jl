using Test
using GaussianMcmc
using DiffEqJump
using Catalyst
using StaticArrays

traj = GaussianMcmc.Trajectory([1.0, 2.0, 3.0], [SA[1,4], SA[2,5], SA[3,6]], [1, 1])
traj_mat = GaussianMcmc.Trajectory([1.0, 2.0, 3.0], [1 2 3; 4 5 6], [1, 1])

@test traj == traj_mat

@test length(traj) == 3 == length(traj.t)
@test collect(traj) == [([1, 4], 1.0, 1), ([2,5], 2.0, 1), ([3, 6], 3.0, 0)]

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

u0 = [50.0]
tspan = (0., 100.)
discrete_prob = DiscreteProblem(u0, tspan)
jump_prob = JumpProblem(sn, discrete_prob, Direct())

integrator = init(jump_prob, SSAStepper(), tstops=Float64[])
ssa_iter = GaussianMcmc.SSAIter(integrator)

traj_sol = GaussianMcmc.collect_trajectory(ssa_iter)
@test length(traj_sol.i) == length(traj_sol) 
sol = integrator.sol
for i in eachindex(sol)
    if i > 1
        @test sol.t[i] == traj_sol.t[i-1]
        @test sol[i] == traj_sol.u[i]
    end
end

for i in 1:length(traj_sol)-1
    if traj_sol.i[i] == 1
        @test (traj_sol.u[i+1] - traj_sol.u[i]) == [-1]
    else
        @test (traj_sol.u[i+1] - traj_sol.u[i]) == [1]
    end
end

# test SSAIterator
iterator = GaussianMcmc.SSAIter(init(jump_prob, SSAStepper()))
collected_values = collect(iterator)
times = getindex.(collected_values, 2)
@test times[begin] > 0.0
@test times[end] == 100.0
@test issorted(times)
@test allunique(times)

traj2 = GaussianMcmc.Trajectory([0.5, 1.5, 2.5], [[1,4], [2,5], [3,6]], [2, 4, 0])

joint = traj |> GaussianMcmc.MergeWith(traj2)

@test collect(joint) == [
    ([1,4,1,4], 0.5, 2),
    ([1,4,2,5], 1.0, 1),
    ([2,5,2,5], 1.5, 4),
    ([2,5,3,6], 2.0, 1),
    ([3,6,3,6], 2.5, 0),
]

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

network = merge(sn, rn)

u0 = [50.0,50.0]
tspan = (0., 100.)
discrete_prob = DiscreteProblem(u0, tspan)
jump_prob = JumpProblem(network, discrete_prob, Direct())
integrator = init(jump_prob, SSAStepper())

partial = GaussianMcmc.sub_trajectory(GaussianMcmc.SSAIter(integrator), [1])
sol = integrator.sol
for i in eachindex(partial.t)
    @test partial.t[i] ∈ vcat(sol.t, 100.0)
    @test partial[i][1] ∈ sol[[1],:]
end


# test trajectory callback
using Transducers
import GaussianMcmc: TrajectoryCallback
import DiffEqBase: DiscreteCallback

cb_traj = GaussianMcmc.Trajectory([30.0, 60.0, 90.0, 100.0], [[100], [110], [120], [130]])
cb = TrajectoryCallback(cb_traj)
cb = DiscreteCallback(cb, cb, save_positions=(false, true))

u0 = [50.0]
tspan = (0., 100.)
discrete_prob = DiscreteProblem(sn, u0, tspan)
jump_prob = JumpProblem(sn, discrete_prob, Direct())
integrator = init(jump_prob, SSAStepper(), callback=cb, tstops=[30.0, 60.0, 90.0])
iter = GaussianMcmc.SSAIter(integrator)

traj = iter |> GaussianMcmc.collect_trajectory

for t in [30, 60, 90] @test t ∈ traj.t end
@test all([[110], [120], [130]] .∈ Ref(traj.u))

sol = integrator.sol
events = traj |> Map((u,t,i)::Tuple -> (u=u,t=t)) |> collect

for i in eachindex(events)
    if i == length(events) continue end
    @test events[i].t ∈ sol.t
    j = searchsortedlast(sol.t, events[i].t)
    @test events[i+1].u == sol.u[j]
end