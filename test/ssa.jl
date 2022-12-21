using Test
import PathWeightSampling as PWS
import Random

# create a simple birth-death system
rates = [1.0, 1.0]
rstoich = [[], [1 => 1]]
nstoich = [[1 => 1], [1 => -1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, 1)

agg = PWS.build_aggregator(PWS.GillespieDirect(), reactions, 1:2; seed=1234)

@test agg.rng == Random.Xoshiro(1234)
@test agg.sumrate == 0.0

agg = PWS.initialize_aggregator(agg, reactions)

@test agg.rng != Random.Xoshiro(1234)
@test agg.u == [0]
@test agg.tstop > 0
@test agg.sumrate == 1.0

trace = PWS.ReactionTrace([], [])

agg2 = PWS.step_ssa(agg, reactions, nothing, trace)

@test agg2.tstop > agg.tstop
@test agg2.u == [1]
@test agg2.sumrate == 2.0
@test agg2.weight != 0

@test trace.rx == [1]
@test trace.t == [agg.tstop]

agg3 = agg
for i = 1:100
    global agg3 = PWS.step_ssa(agg3, reactions, nothing, trace)
end

@test agg3.u[1] == -sum(2 .* trace.rx .- 3)

agg = PWS.initialize_aggregator(agg, reactions, u0=[0], active_reactions=BitSet())

@test agg.tstop == Inf
@test agg.trace_index == 1

trace_new = PWS.ReactionTrace([], [])

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
reactions = PWS.ReactionSet(rates, rstoich, nstoich, 1)

agg = PWS.build_aggregator(PWS.GillespieDirect(), reactions, 1:1)
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

reactions = PWS.ReactionSet(rates, rstoich, nstoich, 2)

u0 = [50, 50]
tspan = (0.0, 10.0)
system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    tspan,
    [0, 0, 1, 2],
    BitSet([3, 4])
)

agg, trace = PWS.generate_trace(system)

ce = agg.weight

@time samples = [PWS.sample(trace, system) for i = 1:10000]

using Statistics
result_direct = mean(ce .- samples)

system_dep = PWS.MarkovJumpSystem(
    PWS.DepGraphDirect(),
    reactions,
    u0,
    tspan,
    [0, 0, 1, 2],
    BitSet([3, 4])
)

agg2, trace2 = PWS.generate_trace(system_dep)

@time samples_dep = [PWS.sample(trace, system_dep) for i = 1:10000]

result_dep = mean(ce .- samples_dep)

@test result_direct ≈ result_dep rtol = 0.01
