using Test
import PathWeightSampling as PWS
import Random
using StaticArrays

# create a simple birth-death system
rates = SA[1.0, 1.0]
rstoich = (SA{Pair{Int, Int}}[], SA[1 => 1])
nstoich = (SA[1 => 1], SA[1 => -1])

reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:X])
reaction_groups = PWS.SSA.make_reaction_groups(reactions, :X)
@test reaction_groups == [1, 2]
@test PWS.SSA.make_reaction_groups(reactions, :S) == [0, 0]

agg = PWS.build_aggregator(PWS.GillespieDirect(), reactions, SA[0], reaction_groups)

@test agg.sumrate == 0.0

agg = PWS.initialize_aggregator(agg, reactions)

@test agg.u == [0]
@test agg.tstop > 0
@test agg.sumrate == 1.0

trace = PWS.ReactionTrace([], [], BitSet(1:2))

agg2 = PWS.step_ssa(agg, reactions, nothing, trace)

@test agg2.tstop > agg.tstop
@test agg2.u == [1]
@test agg2.sumrate == 2.0
@test agg2.weight != 0

@test trace.rx == [1]
@test trace.t == [agg.tstop]

agg3 = agg2
for i = 1:100
    global agg3 = PWS.step_ssa(agg3, reactions, nothing, trace)
end

@test agg3.u[1] == -sum(2 .* trace.rx .- 3)

agg = PWS.initialize_aggregator(agg, reactions, u0=SA[0], active_reactions=BitSet())

@test agg.tstop == Inf
@test agg.trace_index == 1

trace_new = PWS.ReactionTrace([], [], BitSet([1,2]))

agg = PWS.step_ssa(agg, reactions, trace, trace_new)

@test agg.tstop == Inf
@test agg.u == [1]
@test agg.trace_index == 2

for i = 1:100
    global agg = PWS.step_ssa(agg, reactions, trace, trace_new)
    @test agg.tprev == trace.t[i+1]
    @test agg.tstop == Inf
    @test agg.trace_index == i + 2
end

@test trace == trace_new

# test absorbing state

rates = [1.0]
rstoich = [[1 => 1]]
nstoich = [[1 => -1]]
reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:X])

agg = PWS.build_aggregator(PWS.GillespieDirect(), reactions, [0], PWS.SSA.make_reaction_groups(reactions, :X))
agg = PWS.initialize_aggregator(agg, reactions, tspan=(0.0, 10.0))
@test agg.u == [0]
@test agg.tstop == Inf
agg = PWS.step_ssa(agg, reactions, nothing, nothing)
@test agg.tprev == 10.0
@test agg.weight == 0.0

# test MarkovJumpSystem with coupled birth death processes

rates = [50.0, 1.0, 1.0, 1.0]
rstoich = [[], [1 => 1], [1 => 1], [2 => 1]]
nstoich = [[1 => 1], [1 => -1], [2 => 1], [2 => -1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])

u0 = [50, 50]
tspan = (0.0, 10.0)
system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    tspan,
    :S,
    :X
)

rng = Random.Xoshiro(1)
agg, trace = PWS.JumpSystem.generate_trace(system; rng)

@test issorted(trace.t)
@test sort(unique(trace.rx)) ⊆ [1, 2, 3, 4]

rng = Random.Xoshiro(1)
conf = PWS.generate_configuration(system; rng)

@test conf.trace == trace
@test size(conf.traj, 1) == length(conf.species) == length(u0)
@test conf.species == system.reactions.species

df = PWS.to_dataframe(conf)
@test df.time == PWS.discrete_times(system)
@test df.S == conf.traj[1, :]
@test df.X == conf.traj[2, :]

using StaticArrays

rates = SA[50.0, 1.0, 1.0, 1.0]
rstoich = (SA{Pair{Int, Int}}[], SA[1 => 1], SA[1 => 1], SA[2 => 1])
nstoich = (SA[1 => 1], SA[1 => -1], SA[2 => 1], SA[2 => -1])
reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])
u0 = SA[50, 50]
tspan = (0.0, 10.0)
system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    tspan,
    :S,
    :X
)

static_conf = PWS.generate_configuration(system; rng=Random.Xoshiro(1))

@test conf.trace == static_conf.trace
@test conf.traj == static_conf.traj

# ce = agg.weight

# @time samples = [PWS.sample(system, trace)[end] for i = 1:10000]

# using Statistics
# result_direct = mean(ce .- samples)

# system_dep = PWS.MarkovJumpSystem(
#     PWS.DepGraphDirect(),
#     reactions,
#     u0,
#     tspan,
#     [0, 0, 1, 2],
#     BitSet([3, 4])
# )

# agg2, trace2 = PWS.generate_trace(system_dep)

# @time samples_dep = [PWS.sample(system_dep, trace)[end] for i = 1:10000]

# result_dep = mean(ce .- samples_dep)

# @test result_direct ≈ result_dep rtol = 0.01
