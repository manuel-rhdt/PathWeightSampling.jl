using Test
import PathWeightSampling as PWS
import Random
using StaticArrays

# create a simple birth-death system
function make_aggregator(a, b)
    rates = [a, b]
    rstoich = [Pair{Int, Int}[], [1 => 1]]
    nstoich = [[1 => 1], [1 => -1]]
    reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:X])

    reaction_groups = PWS.SSA.make_reaction_groups(reactions, :X)
    @test reaction_groups == [1, 2]
    @test PWS.SSA.make_reaction_groups(reactions, :S) == [0, 0]

    reactions, PWS.build_aggregator(PWS.GillespieDirect(), reactions, [0], reaction_groups, seed=1)
end

reactions, agg = make_aggregator(1.0, 1.0)
@test agg.rng == Random.Xoshiro(1)
@test agg.sumrate == 0.0

PWS.initialize_aggregator!(agg, reactions)
@test agg.u == [0]
@test agg.tstop == Random.randexp(Random.Xoshiro(1))
@test agg.sumrate == 1.0

trace = PWS.ReactionTrace([], [], BitSet(1:2))

agg2 = PWS.SSA.step_ssa!(copy(agg), reactions, nothing, trace)

@test agg2.tstop > agg.tstop
@test agg2.u == [1]
@test agg2.sumrate == 2.0
@test agg2.weight != 0

@test trace.rx == [1]
@test trace.t == [agg.tstop]

agg3 = agg2
for i = 1:100
    PWS.SSA.step_ssa!(agg3, reactions, nothing, trace)
end

@test agg3.u[1] == -sum(2 .* trace.rx .- 3)

PWS.SSA.initialize_aggregator!(agg, reactions, u0=SA[0], active_reactions=BitSet())

@test agg.tstop == Inf
@test agg.trace_index == 1

trace_new = PWS.ReactionTrace([], [], BitSet([1,2]))

PWS.SSA.step_ssa!(agg, reactions, trace, trace_new)

@test agg.tstop == Inf
@test agg.u == [1]
@test agg.trace_index == 2

for i = 1:100
    PWS.SSA.step_ssa!(agg, reactions, trace, trace_new)
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
PWS.SSA.initialize_aggregator!(agg, reactions, tspan=(0.0, 10.0))
@test agg.u == [0]
@test agg.tstop == Inf
PWS.SSA.step_ssa!(agg, reactions, nothing, nothing)
@test agg.tprev == 10.0
@test agg.weight == 0.0

# test MarkovJumpSystem with coupled birth death processes

rates = [50.0, 1.0, 1.0, 1.0]
rstoich = [Pair{Int, Int}[], [1 => 1], [1 => 1], [2 => 1]]
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

agg, trace = PWS.JumpSystem.generate_trace(system; rng=Random.Xoshiro(1))
_, trace2 = PWS.JumpSystem.generate_trace(system; rng=Random.Xoshiro(1))
_, trace3 = PWS.JumpSystem.generate_trace(system; rng=Random.Xoshiro(2))

@test trace == trace2
@test trace != trace3

@test issorted(trace.t)
@test sort(unique(trace.rx)) ⊆ [1, 2, 3, 4]

conf = PWS.generate_configuration(system; rng=Random.Xoshiro(1))

@test conf.trace == trace
@test size(conf.traj, 1) == length(conf.species) == length(u0)
@test conf.species == system.reactions.species

df = PWS.to_dataframe(conf)
@test df.time == PWS.discrete_times(system)
@test df.S == conf.traj[1, :]
@test df.X == conf.traj[2, :]

using StaticArrays

rates = SA[50.0, 1.0, 1.0, 1.0]
rstoich = SA[Pair{Int, Int}[], [1 => 1], [1 => 1], [2 => 1]]
nstoich = SA[[1 => 1], [1 => -1], [2 => 1], [2 => -1]]
reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])
u0 = SA[50.0, 50.0]
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

κ, λ, ρ, μ = rates


result_object = PWS.mutual_information(system, PWS.SMCEstimate(256), num_samples=1000)

using DataFrames, Statistics
@show median(result_object.metadata.CPUTime)

sem(x) = sqrt(var(x) / length(x))
pws_result = combine(
    groupby(result_object.result, :time), 
    :MutualInformation => mean => :MI,
    :MutualInformation => sem => :Err
)

relative_error = pws_result.Err[end] / pws_result.MI[end]
@test relative_error < 5e-2

rate_estimate = (pws_result.MI[end] - pws_result.MI[3]) / (pws_result.time[end] - pws_result.time[3])
rate_analytical = λ / 2 * (sqrt(1 + 2ρ / λ) - 1) # From Moor et al. PRR 2023

@test rate_estimate ≈ rate_analytical rtol=5*relative_error
