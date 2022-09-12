import ModelingToolkit
import ModelingToolkit: build_function, substitute
import Catalyst
import Catalyst: ReactionSystem
using StaticArrays
import Base.show
using Setfield

abstract type AbstractJumpSet end

struct ReactionSet <: AbstractJumpSet
    rates::Vector{Float64}
    rstoich::Vector{Vector{Pair{Int64,Int64}}}
    nstoich::Vector{Vector{Pair{Int64,Int64}}}
    nspecies::Int
end

num_reactions(rs::AbstractJumpSet) = length(rs.rates)
num_species(rs::AbstractJumpSet) = rs.nspecies

struct JumpSet{Jumps} <: AbstractJumpSet
    reactions::ReactionSet
    jumps::Jumps
end

num_reactions(js::JumpSet) = num_reactions(js.reactions) + num_reactions(js.jumps)
num_species(js::JumpSet) = num_species(js.reactions)

abstract type Trace end

"""
A (partial) record of a SSA execution.

Contains a list of reaction firing times together with a list of the
corresponding reaction indices. A full reaction trace contains every reaction
that fired during the SSA. A partial trace only records reactions involving a
specific species.
"""
struct ReactionTrace <: Trace
    "a vector of reaction times"
    t::Vector{Float64}

    "a vector of reaction indices"
    rx::Vector{Int16}
end

ReactiontTrace() = ReactionTrace([], [])

function Base.empty!(trace::ReactionTrace)
    empty!(trace.t)
    empty!(trace.rx)
    trace
end

Base.:(==)(t1::ReactionTrace, t2::ReactionTrace) = (t1.t == t2.t && t1.rx == t2.rx)

struct HybridTrace{U} <: Trace
    "a vector of reaction times"
    t::Vector{Float64}

    "a vector of reaction indices"
    rx::Vector{Int16}

    "a vector of external signal"
    u::U

    "sampling interval of external trajectory"
    dt::Float64
end

abstract type AbstractJumpRateAggregatorAlgorithm end

struct GillespieDirect <: AbstractJumpRateAggregatorAlgorithm end

struct DepGraphDirect <: AbstractJumpRateAggregatorAlgorithm end


abstract type AbstractJumpRateAggregator end

"""
A jump rate aggregator keeps track of the current state of a SSA simulation and can simulataneously
compute the trajectory likelihood.

To compute the conditional probability of a component's trajectory given another component's trajectory,
we can first record and later "replay" the reaction trace of a simulation.

# Recording a reaction trace

A reaction trace is a time trace of reaction indices and the corresponding firing times. 

During the simulation, every time a reaction fires, the corresponding reaction index is recorded alongside
the simulation time.

We can record a reaction trace of only a subset of reactions specified by `traced_reactions`.

# Conditional SSA simulation (replaying a recorded trace)

We can perform a SSA simulation conditional on an existing time trace. All reactions recorded in the time
trace are replayed during the simulation, while non-recorded reactions will still occur stochastically. This
allows us to evaluate the conditional likelihood of one component's trajectory given another component's
trajectory.

# Computing the log-likelihood of the simulated trajectory

During the simulation we can evaluate the likelihood of the produced trajectory on the fly. The log-likelihood
is stored in the aggregator's `weight` field.

Often we are only interested in the likelihood of a particular component's trajectory conditioned on all the
other components' trajectories. This is made possible by the concept of a *group* of reactions. A *group* of reactions
should be interpreted as a set of reactions whose effect is indistinguishible to a downstream observer.
For example, suppose that downstream chemical reactions can only read out the component `X` of the
reaction system. Then, all reactions that modify component `X` in the same way belong to the same group.

The jump rate aggregator keeps track of the combined propensities of each reaction group. Every time a
reaction from the group fires, the log-likelihood is updated by the log of the group's propensity. Similarly,
every time a reaction fires that is not part of any group, the log-likelihood is updated by the sum of all
groups' propensities (to account for the waiting time probability).
"""
struct DirectAggregator{U,Map} <: AbstractJumpRateAggregator
    "The current state vector"
    u::U

    "The sum of all reaction propensities. This is used to compute the waiting time
    before the next reaction."
    sumrate::Float64

    "A vector of the current reaction propensities for every reaction."
    rates::Vector{Float64}

    "the sum of all groups' propensities"
    gsumrate::Float64

    "the individual groups' propensities"
    grates::Vector{Float64}

    "set of reaction indices that are allowed to fire"
    active_reactions::BitSet

    "set of reaction indices that will be saved into the trace"
    traced_reactions::BitSet

    "maps reaction indices to group indices"
    ridtogroup::Map

    "time span for aggregation"
    tspan::Tuple{Float64,Float64}

    "time of last recorded reaction"
    tprev::Float64

    "time of next stochastic reaction"
    tstop::Float64

    "current trace index"
    trace_index::Int

    "accumulated log-probability"
    weight::Float64
end

function build_aggregator(alg::GillespieDirect, reactions::AbstractJumpSet, ridtogroup, tspan=(0.0, Inf64))
    ngroups = maximum(ridtogroup)
    nreactions = num_reactions(reactions)
    nspecies = num_species(reactions)
    DirectAggregator(
        zeros(Int64, nspecies),
        0.0, zeros(nreactions),
        0.0, zeros(ngroups),
        BitSet(1:nreactions),
        BitSet(1:nreactions),
        ridtogroup,
        tspan,
        tspan[1],
        tspan[1],
        1,
        0.0)
end

struct DepGraphAggregator{U,Map,DepGraph} <: AbstractJumpRateAggregator
    "The current state vector"
    u::U

    "The sum of all reaction propensities. This is used to compute the waiting time
    before the next reaction."
    sumrate::Float64

    "A vector of the current reaction propensities for every reaction."
    rates::Vector{Float64}

    "the sum of all groups' propensities"
    gsumrate::Float64

    "the individual groups' propensities"
    grates::Vector{Float64}

    "set of reaction indices that are allowed to fire"
    active_reactions::BitSet

    "set of reaction indices that will be saved into the trace"
    traced_reactions::BitSet

    "maps reaction indices to group indices"
    ridtogroup::Map

    "time span for aggregation"
    tspan::Tuple{Float64,Float64}

    "time of last recorded reaction"
    tprev::Float64

    "time of next stochastic reaction"
    tstop::Float64

    "current trace index"
    trace_index::Int

    "accumulated log-probability"
    weight::Float64

    "dependency graph"
    depgraph::DepGraph

    "optimized (sorted by rate) order to search for next jump"
    jump_search_order::Vector{Int}
end

function build_aggregator(alg::DepGraphDirect, reactions::AbstractJumpSet, ridtogroup, tspan=(0.0, Inf64))
    ngroups = maximum(ridtogroup)
    nreactions = num_reactions(reactions)
    nspecies = num_species(reactions)
    depgraph = make_depgraph(reactions)
    DepGraphAggregator(
        zeros(Int64, nspecies),
        0.0, zeros(nreactions),
        0.0, zeros(ngroups),
        BitSet(1:nreactions),
        BitSet(1:nreactions),
        ridtogroup,
        tspan,
        tspan[1],
        tspan[1],
        1,
        0.0,
        depgraph,
        collect(1:nreactions)
    )
end

function initialize_aggregator(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet;
    u0=agg.u,
    tspan=(0.0, Inf64),
    active_reactions=agg.active_reactions,
    traced_reactions=agg.traced_reactions)
    agg = @set agg.u = u0
    agg = @set agg.active_reactions = active_reactions
    agg = @set agg.traced_reactions = traced_reactions
    agg = @set agg.tspan = tspan
    agg = @set agg.tprev = tspan[1]
    agg = @set agg.weight = 0.0
    agg = @set agg.sumrate = 0.0
    agg = @set agg.gsumrate = 0.0
    agg = @set agg.trace_index = 1
    agg = update_rates(agg, reactions)
    agg = @set agg.tstop = tspan[1] + randexp() / agg.sumrate
    agg
end

function initialize_aggregator(
    agg::DepGraphAggregator,
    reactions::AbstractJumpSet;
    u0=agg.u,
    tspan=(0.0, Inf64),
    active_reactions=agg.active_reactions,
    traced_reactions=agg.traced_reactions)
    agg = @set agg.u = u0
    agg = @set agg.active_reactions = active_reactions
    agg = @set agg.traced_reactions = traced_reactions
    agg = @set agg.tspan = tspan
    agg = @set agg.tprev = tspan[1]
    agg = @set agg.weight = 0.0
    agg = @set agg.sumrate = 0.0
    agg = @set agg.gsumrate = 0.0
    agg = @set agg.trace_index = 1
    agg = update_rates(agg, reactions)
    agg = @set agg.tstop = tspan[1] + randexp() / agg.sumrate
    agg = @set agg.jump_search_order = collect(active_reactions)
    agg
end

set_tspan(agg::AbstractJumpRateAggregator, tspan) = @set agg.tspan = tspan


function ReactionSet(js::ModelingToolkit.JumpSystem, p)
    parammap = map(Pair, ModelingToolkit.parameters(js), p)
    statetoid = Dict(ModelingToolkit.value(state) => i for (i, state) in enumerate(ModelingToolkit.states(js)))

    rates = Float64[]
    rstoich_vec = Vector{Pair{Int64,Int64}}[]
    nstoich_vec = Vector{Pair{Int64,Int64}}[]

    for eq in ModelingToolkit.equations(js)
        rate = ModelingToolkit.value(ModelingToolkit.substitute(eq.scaled_rates, parammap))
        rstoich = sort!([statetoid[ModelingToolkit.value(spec)] => stoich for (spec, stoich) in eq.reactant_stoch])
        nstoich = sort!([statetoid[ModelingToolkit.value(spec)] => stoich for (spec, stoich) in eq.net_stoch])

        push!(rates, rate)
        push!(rstoich_vec, rstoich)
        push!(nstoich_vec, nstoich)
    end

    ReactionSet(rates, rstoich_vec, nstoich_vec, length(ModelingToolkit.states(js)))
end

function species_to_dependent_reaction_map(reactions::AbstractJumpSet)
    nspecies = num_species(reactions)
    # map from a species to reactions that depend on it
    spec_to_dep_rxs = [Vector{Int}() for n = 1:nspecies]
    for (rx, complex) in enumerate(reactions.rstoich)
        for (spec, stoch) in complex
            push!(spec_to_dep_rxs[spec], rx)
        end
    end

    foreach(s -> unique!(sort!(s)), spec_to_dep_rxs)
    spec_to_dep_rxs
end

function species_to_dependent_reaction_map(js::JumpSet)
    map_reactions = species_to_dependent_reaction_map(js.reactions)
    nreactions = num_reactions(js.reactions)
    map_jumps = species_to_dependent_reaction_map(js.jumps)
    for spec in keys(map_reactions)
        if haskey(map_jumps, spec)
            append!(map_reactions[spec], map_jumps[spec] .+ nreactions)
        end
    end

    foreach(s -> unique!(sort!(s)), map_reactions)
    map_reactions
end

mutated_species(rs::AbstractJumpSet, index) = (spec for (spec, stoch) in rs.nstoich[index])
function mutated_species(js::JumpSet, index)
    nreactions = num_reactions(js.reactions)
    if index <= nreactions
        mutated_species(js.reactions, index)
    else
        mutated_species(js.jumps, index - nreactions)
    end
end

function make_depgraph(reactions::AbstractJumpSet)
    nreactions = num_reactions(reactions)
    spec_to_dep_rxs = species_to_dependent_reaction_map(reactions)

    # create map from rx to reactions depending on it
    dep_graph = [Vector{Int}() for n = 1:nreactions]
    for rx in 1:nreactions
        # rx changes spec, hence rxs depending on spec depend on rx
        for spec in mutated_species(reactions, rx)
            for dependent_rx in spec_to_dep_rxs[spec]
                push!(dep_graph[rx], dependent_rx)
            end
        end
    end

    foreach(deps -> unique!(sort!(deps)), dep_graph)
    dep_graph
end

struct TrajectoryDistribution{A}
    reactions::ReactionSet
    aggregator::A
end

function TrajectoryDistribution(reactions::ReactionSet, alg::AbstractJumpRateAggregatorAlgorithm, ridtogroup=1:length(reactions.rates))
    agg = build_aggregator(alg, reactions, ridtogroup)
    TrajectoryDistribution(reactions, agg)
end

distribution(rn::ReactionSystem, p; update_map=1:Catalyst.numreactions(rn), alg=GillespieDirect()) = TrajectoryDistribution(ReactionSet(convert(ModelingToolkit.JumpSystem, rn), p), alg, update_map)

@fastmath function Distributions.logpdf(dist::TrajectoryDistribution, trajectory)::Float64
    traj_iter = trajectory_iterator(trajectory)
    tprev = 0.0
    result = 0.0
    agg = initialize_aggregator(dist.aggregator)
    for (u, t, i) in traj_iter
        agg = update_rates(agg, dist.reactions)

        dt = t - tprev
        result -= dt * agg.sumrate
        if i != 0
            @inbounds gid = agg.ridtogroup[i]
            gid != 0 && @inbounds result += log(agg.gsums[gid])
        end

        tprev = t
    end

    result
end

@fastmath function fold_logpdf(dist::TrajectoryDistribution, agg::AbstractJumpRateAggregator, (u, t, i))
    agg = update_rates(agg, dist.reactions)
    dt = t - agg.tprev
    log_jump_prob = 0.0
    if i != 0
        @inbounds gid = agg.ridtogroup[i]
        gid != 0 && @inbounds log_jump_prob = log(agg.gsums[gid])
    end
    log_surv_prob = -dt * agg.sumrate
    agg = @set agg.weight = agg.weight + log_surv_prob + log_jump_prob
    agg = @set agg.tprev = t
end

function step_ssa(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet,
    trace::Union{Nothing,<:Trace},
    out_trace::Union{Nothing,<:Trace};
)
    tindex = agg.trace_index

    if (!isnothing(trace)) && tindex <= length(trace.t) && agg.tstop >= trace.t[tindex]
        # perform reaction from pre-recorded trace

        tnow = trace.t[tindex]
        if tnow >= agg.tspan[2]
            tnow = agg.tspan[2]
            return update_weight(agg, tnow)
        end

        # perform reaction from trace
        srate1 = agg.sumrate # sumrate before jump
        rx = trace.rx[tindex]

        # execute reaction
        agg = update_weight(agg, tnow, rx)
        executerx!(agg.u, rx, reactions)
        agg = update_rates(agg, reactions, rx)
        srate2 = agg.sumrate # sumrate after jump

        # update tstop, taking into account the change of total propensity
        t1 = agg.tstop
        agg = @set agg.tstop = t1 == Inf ? Inf : (srate1 * t1 + (srate2 - srate1) * tnow) / srate2

        # advance trace
        agg = @set agg.trace_index = tindex + 1

    else
        # perform stochastic reaction

        tnow = agg.tstop
        if agg.tstop >= agg.tspan[2]
            tnow = agg.tspan[2]
            return update_weight(agg, tnow)
        end

        # perform reaction and update rates
        rx = select_reaction(agg)
        agg = update_weight(agg, tnow, rx)
        executerx!(agg.u, rx, reactions)
        agg = update_rates(agg, reactions, rx)

        # draw next tstop
        new_tstop = tnow + randexp() / agg.sumrate
        agg = @set agg.tstop = new_tstop
    end

    # store reaction event in trace
    if (!isnothing(out_trace)) && rx ∈ agg.traced_reactions
        push!(out_trace.rx, rx)
        push!(out_trace.t, tnow)
    end

    agg
end

@inline @fastmath function select_reaction(agg::DirectAggregator)
    rx = 0
    x = rand() * agg.sumrate
    @inbounds for reaction in agg.active_reactions
        rx = reaction
        if x <= agg.rates[rx]
            break
        end
        x -= agg.rates[rx]
    end
    rx
end

@inline @fastmath function select_reaction(agg::DepGraphAggregator)
    rx = 0
    jsidx = 1
    x = rand() * agg.sumrate
    @inbounds for (i, reaction) in enumerate(agg.jump_search_order)
        rx = reaction
        jsidx = i
        if x <= agg.rates[rx]
            break
        end
        x -= agg.rates[rx]
    end

    # update jump order
    jso = agg.jump_search_order
    if jsidx != 1
        @inbounds tmp = jso[jsidx]
        @inbounds jso[jsidx] = jso[jsidx-1]
        @inbounds jso[jsidx-1] = tmp
    end

    rx
end

@inline @fastmath function update_weight(agg::AbstractJumpRateAggregator, tnow, rx=nothing)
    # compute log probability of jump; update weight
    Δt = tnow - agg.tprev
    log_jump_prob = 0.0
    if !isnothing(rx)
        @inbounds gid = agg.ridtogroup[rx]
        gid != 0 && @inbounds log_jump_prob += log(agg.grates[gid])
    end
    agg = @set agg.weight = agg.weight + log_jump_prob - Δt * agg.gsumrate
    agg = @set agg.tprev = tnow
end

# Code adapted from JumpProcesses.jl
@inline @fastmath function executerx!(speciesvec::AbstractVector, rxidx::Integer, reactions::AbstractJumpSet)
    @inbounds net_stoch = reactions.nstoich[rxidx]
    @inbounds for specstoch in net_stoch
        speciesvec[specstoch[1]] += specstoch[2]
    end
    nothing
end

@inline @fastmath function executerx!(speciesvec::AbstractVector, rxidx::Integer, js::JumpSet)
    nreactions = num_reactions(js.reactions)
    if rxidx <= nreactions
        executerx!(speciesvec, rxidx, js.reactions)
    else
        executerx!(speciesvec, rxidx - nreactions, js.jumps)
    end
end

function step_energy(dist::TrajectoryDistribution, agg::AbstractJumpRateAggregator, (u, t, i))
    if t <= agg.tspan[1]
        return agg
    end
    if t > agg.tspan[2]
        if agg.tprev >= agg.tspan[2]
            return agg
        end
        fold_logpdf(dist, agg, (u, agg.tspan[2], 0))
    else
        fold_logpdf(dist, agg, (u, t, i))
    end
end

function trajectory_energy(dist::TrajectoryDistribution, traj; tspan=(0.0, Inf64))
    agg = initialize_aggregator(dist.aggregator, tspan=tspan)
    traj_iter = trajectory_iterator(traj)
    agg = Base.foldl((acc, x) -> step_energy(dist, acc, x), traj_iter; init=agg)
    agg.weight
end

function cumulative_logpdf!(result::AbstractVector, dist::TrajectoryDistribution, traj, dtimes::AbstractVector)
    tspan = (first(dtimes), last(dtimes))
    agg = initialize_aggregator(dist.aggregator, tspan=tspan)
    result[1] = zero(eltype(result))
    traj_iter = trajectory_iterator(traj)
    result_agg, k = Base.foldl(traj_iter; init=(agg, 1)) do (agg, k), (u, t, i)
        if t <= agg.tspan[1]
            return agg, k
        end

        t = min(t, agg.tspan[2])
        agg = update_rates(agg, dist.reactions)

        tprev = agg.tprev
        while k <= length(dtimes) && dtimes[k] < t
            result[k] -= (dtimes[k] - tprev) * agg.sumrate
            tprev = dtimes[k]
            k += 1
            result[k] = result[k-1]
        end
        result[k] -= (t - tprev) * agg.sumrate

        log_jump_prob = 0.0
        if i != 0
            @inbounds gid = agg.ridtogroup[i]
            gid != 0 && @inbounds log_jump_prob = log(agg.gsums[gid])
        end
        result[k] += log_jump_prob

        agg = @set agg.weight = agg.weight - (t - agg.tprev) * agg.sumrate + log_jump_prob
        agg = @set agg.tprev = t
        agg, k
    end

    result
end

cumulative_logpdf(dist::TrajectoryDistribution, trajectory, dtimes::AbstractVector) = cumulative_logpdf!(zeros(length(dtimes)), dist, trajectory, dtimes)


@inline @fastmath function evalrxrate(speciesvec::AbstractVector{T}, rxidx::Int64, rs::ReactionSet) where {T}
    val = Float64(1.0)
    @inbounds for specstoch in rs.rstoich[rxidx]
        @inbounds specpop = speciesvec[specstoch[1]]
        val *= Float64(specpop)
        @inbounds for k = 2:specstoch[2]
            specpop -= one(specpop)
            val *= Float64(specpop)
        end
    end

    @inbounds val * rs.rates[rxidx]
end

@inline @fastmath function evalrxrate(speciesvec::AbstractVector{T}, rxidx::Int64, js::JumpSet) where {T}
    nreactions = num_reactions(js.reactions)
    if rxidx <= nreactions
        evalrxrate(speciesvec, rxidx, js.reactions)
    else
        evalrxrate(speciesvec, rxidx - nreactions, js.jumps)
    end
end

@inline function update_reaction_rate!(aggregator::AbstractJumpRateAggregator, reactions::AbstractJumpSet, rx::Int)
    rate = evalrxrate(aggregator.u, rx, reactions)
    @inbounds gid = aggregator.ridtogroup[rx]
    if gid != 0
        @inbounds aggregator.grates[gid] += -aggregator.rates[rx] + rate
    end
    @inbounds aggregator.rates[rx] = rate
    rate
end

@fastmath function update_rates(aggregator::DirectAggregator, reactions::AbstractJumpSet, prev_reaction::Integer=0)
    sumrate = zero(aggregator.sumrate)
    for rx in 1:num_reactions(reactions)
        update_reaction_rate!(aggregator, reactions, rx)
    end

    for rx in aggregator.active_reactions
        sumrate += aggregator.rates[rx]
    end

    aggregator = @set aggregator.gsumrate = sum(aggregator.grates)
    aggregator = @set aggregator.sumrate = sumrate
end

@fastmath function update_rates(aggregator::DepGraphAggregator, reactions::AbstractJumpSet, prev_reaction::Integer=0)
    sumrate = aggregator.sumrate
    if prev_reaction == 0
        # recompute all rates
        for rx in 1:num_reactions(reactions)
            update_reaction_rate!(aggregator, reactions, rx)
        end
        sumrate = zero(sumrate)
        for rx in aggregator.active_reactions
            sumrate += aggregator.rates[rx]
        end
    else
        # only recompute dependent rates
        @inbounds for rx in aggregator.depgraph[prev_reaction]
            if rx ∈ aggregator.active_reactions
                sumrate -= aggregator.rates[rx]
                rate = update_reaction_rate!(aggregator, reactions, rx)
                sumrate += rate
            else
                update_reaction_rate!(aggregator, reactions, rx)
            end
        end
    end

    aggregator = @set aggregator.gsumrate = sum(aggregator.grates)
    aggregator = @set aggregator.sumrate = sumrate
end
