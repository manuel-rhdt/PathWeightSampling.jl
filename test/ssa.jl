using Test
using PathWeightSampling

# create a simple birth-death system
rates = [1.0, 1.0]
rstoich = [[], [1 => 1]]
nstoich = [[1 => 1], [1 => -1]]

reactions = PathWeightSampling.ReactionSet(rates, rstoich, nstoich, 1)

agg = PathWeightSampling.build_aggregator(PathWeightSampling.GillespieDirect(), reactions, 1:2; seed=1234)

@test agg.rng == Xoshiro(1234)
@test agg.sumrate == 0.0

agg = PathWeightSampling.initialize_aggregator(agg, reactions)

@test agg.rng != Xoshiro(1234)
@test agg.u == [0]
@test agg.tstop > 0
@test agg.sumrate == 1.0

trace = PathWeightSampling.ReactionTrace([], [])

agg2 = PathWeightSampling.step_ssa(agg, reactions, nothing, trace)

@test agg2.tstop > agg.tstop
@test agg2.u == [1]
@test agg2.sumrate == 2.0
@test agg2.weight != 0

@test trace.rx == [1]
@test trace.t == [agg.tstop]

agg3 = agg
for i = 1:100
    agg3 = PathWeightSampling.step_ssa(agg3, reactions, nothing, trace)
end

trace

@test agg3.u[1] == -sum(2 .* trace.rx .- 3)

agg = PathWeightSampling.initialize_aggregator(agg, reactions, u0=[0], active_reactions=BitSet())

@test agg.tstop == Inf
@test agg.trace_index == 1

trace_new = PathWeightSampling.ReactionTrace([], [])

agg = PathWeightSampling.step_ssa(agg, reactions, trace, trace_new)

@test agg.tstop == Inf
@test agg.u == [1]
@test agg.trace_index == 2

for i = 1:100
    agg = PathWeightSampling.step_ssa(agg, reactions, trace, trace_new)
    @test agg.tprev == trace.t[i+1]
    @test agg.tstop == Inf
    @test agg.trace_index == i + 2
end

@test trace == trace_new

# test absorbing state

rates = [1.0]
rstoich = [[1 => 1]]
nstoich = [[1 => -1]]
reactions = PathWeightSampling.ReactionSet(rates, rstoich, nstoich, 1)

agg = PathWeightSampling.build_aggregator(PathWeightSampling.GillespieDirect(), reactions, 1:1)
agg = PathWeightSampling.initialize_aggregator(agg, reactions, tspan=(0.0, 10.0))
@test agg.u == [0]
@test agg.tstop == Inf
agg = PathWeightSampling.step_ssa(agg, reactions, nothing, nothing)
@test agg.tprev == 10.0
@test agg.weight == 0.0

# test MarkovJumpSystem with coupled birth death processes

rates = [50.0, 1.0, 1.0, 1.0]
rstoich = [[], [1 => 1], [1 => 1], [2 => 1]]
nstoich = [[1 => 1], [1 => -1], [2 => 1], [2 => -1]]

reactions = PathWeightSampling.ReactionSet(rates, rstoich, nstoich, 2)

system = PathWeightSampling.MarkovJumpSystem(
    PathWeightSampling.GillespieDirect(),
    reactions,
    [0, 0, 1, 2],
    BitSet([3, 4])
)

agg, trace = PathWeightSampling.generate_trace(system, [50, 50], (0.0, 10.0))

ce = agg.weight

@time samples = [PathWeightSampling.sample(trace, system, [50, 50], (0.0, 10.0)) for i = 1:10000]

using Statistics
result_direct = mean(ce .- samples)

system_dep = PathWeightSampling.MarkovJumpSystem(
    PathWeightSampling.DepGraphDirect(),
    reactions,
    [0, 0, 1, 2],
    BitSet([3, 4])
)

agg2, trace2 = PathWeightSampling.generate_trace(system_dep, [50, 50], (0.0, 10.0))

@time samples_dep = [PathWeightSampling.sample(trace, system_dep, [50, 50], (0.0, 10.0)) for i = 1:10000]

result_dep = mean(ce .- samples_dep)

@test result_direct ≈ result_dep rtol = 0.05

# JumpSet tests

struct HillReactions <: PathWeightSampling.AbstractJumpSet
    rates::Vector{Float64}
    hill_coeffs::Vector{Float64}
    diss_consts::Vector{Float64}
    rstoich::Vector{Vector{Pair{Int64,Int64}}}
    nstoich::Vector{Vector{Pair{Int64,Int64}}}
    nspecies::Int64
end

function PathWeightSampling.evalrxrate(speciesvec::AbstractVector, rxidx::Int64, hr::HillReactions)
    val = Float64(1.0)
    @inbounds for specstoch in hr.rstoich[rxidx]
        @inbounds specpop = Float64(speciesvec[specstoch[1]])
        @inbounds n = hr.hill_coeffs[rxidx]
        @inbounds K = hr.diss_consts[rxidx]
        val *= specpop^n / (K + specpop^n)
    end

    @inbounds val * hr.rates[rxidx]
end

hr = HillReactions(
    [1.0],
    [5.0],
    [5.0],
    [[1 => 1]],
    [[2 => 1]],
    2
)

@test PathWeightSampling.num_reactions(hr) == 1
@test PathWeightSampling.num_species(hr) == 2

@test PathWeightSampling.make_depgraph(hr) == [[]]

# ChemotaxisJumps

system = PathWeightSampling.simple_chemotaxis_system(duration=20.0, dt=0.05)
@test sum(system.u0[system.reactions.receptors]) == 25

system.agg.cache

PathWeightSampling.make_depgraph(system.reactions)

for (rid, gid) in enumerate(system.agg.ridtogroup)
    if gid == 1
        @test PathWeightSampling.reaction_type(system.reactions, rid)[1] == 2
    elseif gid == 2
        @test PathWeightSampling.reaction_type(system.reactions, rid)[1] == 3
    else
        @test PathWeightSampling.reaction_type(system.reactions, rid)[1] < 2
    end
end

conf = PathWeightSampling.generate_configuration(system)
conf.traj
system.agg.cache.p_a

@test conf.traj[1, 2:end] == conf.trace.u

agg, trace = PathWeightSampling.generate_trace(system)
agg.cache
@time agg, trace = PathWeightSampling.generate_trace(system)

agg.jump_search_order

alg = SMCEstimate(128)

@time mi = PathWeightSampling.mutual_information(system, alg)
mi.MutualInformation
mi.Trajectory