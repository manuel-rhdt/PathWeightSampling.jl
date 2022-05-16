using Test
using PathWeightSampling
using DiffEqJump
using Catalyst
using StaticArrays

traj = PathWeightSampling.Trajectory([SA[1, 4], SA[2, 5], SA[3, 6]], [1.0, 2.0, 3.0], [1, 1, 0])
traj_mat = PathWeightSampling.Trajectory([1 2 3; 4 5 6], [1.0, 2.0, 3.0], [1, 1, 0])

@test traj == traj_mat
@test length(traj) == 3 == length(traj.t)
@test traj[1] == [1, 4] == traj[:, 1]
@test traj[1, :] == [1, 2, 3]
@test traj[2, :] == [4, 5, 6]
@test traj[1:2] == PathWeightSampling.Trajectory([SA[1, 4], SA[2, 5]], [1.0, 2.0], [1, 1])
@test traj[:, :] == [1 2 3; 4 5 6]

@test collect(tuples(traj)) == [([1, 4], 1.0, 1), ([2, 5], 2.0, 1), ([3, 6], 3.0, 0)]
@test traj(0.0) == [1, 4]
@test traj(1.5) == [2, 5]
@test traj(2.9) == [3, 6]

@test traj([-1.0, 0.0, 0.5, 1.5, 3.5]) == [1 1 1 2 3; 4 4 4 5 6]

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

u0 = [50.0]
tspan = (0.0, 100.0)
discrete_prob = DiscreteProblem(sn, u0, tspan, [])
jump_prob = JumpProblem(sn, discrete_prob, Direct())

integrator = init(jump_prob, SSAStepper(), tstops=Float64[])
ssa_iter = PathWeightSampling.SSAIter(integrator)

traj_sol = PathWeightSampling.collect_trajectory(ssa_iter)

@test length(traj_sol.i) == length(traj_sol)
@test traj_sol.syms == [Symbol("S(t)")]

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
iterator = PathWeightSampling.SSAIter(init(jump_prob, SSAStepper()))
collected_values = collect(iterator)
times = getindex.(collected_values, 2)
@test times[begin] > 0.0
@test times[end] == 100.0
@test issorted(times)
@test allunique(times)

traj2 = PathWeightSampling.Trajectory([[1, 4], [2, 5], [3, 6]], [0.5, 1.5, 2.5], [2, 4, 0])

joint = PathWeightSampling.merge_trajectories(traj, traj2)
@test collect(joint) == [
    ([1, 4, 1, 4], 0.5, 2),
    ([1, 4, 2, 5], 1.0, 1),
    ([2, 5, 2, 5], 1.5, 4),
    ([2, 5, 3, 6], 2.0, 1),
    ([3, 6, 3, 6], 2.5, 0),
]

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅
end

network = ModelingToolkit.extend(rn, sn)

u0 = [50.0, 50.0]
tspan = (0.0, 100.0)
discrete_prob = DiscreteProblem(network, u0, tspan, [])
jump_prob = JumpProblem(network, discrete_prob, Direct())

iter = PathWeightSampling.SSAIter(init(jump_prob, SSAStepper(), seed=1))
full = PathWeightSampling.collect_trajectory(iter)
iter = PathWeightSampling.SSAIter(init(jump_prob, SSAStepper(), seed=1))
partial = PathWeightSampling.collect_sub_trajectory(iter, [1])

@test full.syms == [Symbol("S(t)"), Symbol("X(t)")]
@test partial.syms == [Symbol("S(t)")]

sol = iter.integrator.sol
for i in eachindex(partial.t)
    @test partial.t[i] ∈ vcat(sol.t, 100.0)
    @test partial.u[i][1] ∈ sol[[1], :]
    t = partial.t[i]
    @test partial(t) == partial.u[min(i + 1, length(partial))]
    @test partial(t) == full(t)[[1]]
end


# test driven jump problem
import PathWeightSampling
using PathWeightSampling: DrivenJumpProblem

cb_traj = PathWeightSampling.Trajectory([[50], [110], [120], [130]], [30.0, 60.0, 90.0, 100.0])
u0 = [50.0]
tspan = (0.0, 100.0)
discrete_prob = DiscreteProblem(sn, u0, tspan, [])
jump_prob = JumpProblem(sn, discrete_prob, Direct())
djp = DrivenJumpProblem(jump_prob, cb_traj)
integrator = init(djp)
iter = PathWeightSampling.SSAIter(integrator)
traj = iter |> PathWeightSampling.collect_trajectory

for t in [30, 60, 90]
    @test t ∈ traj.t
end
@test traj(30.0)[1] == 110
@test traj(60.0)[1] == 120
@test traj(90.0)[1] == 130
