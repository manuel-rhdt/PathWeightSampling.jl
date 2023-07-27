module SSA

export AbstractJumpSet, ReactionSet, AbstractJumpRateAggregator, AbstractJumpRateAggregatorAlgorithm
export DirectAggregator, DepGraphAggregator, GillespieDirect, DepGraphDirect, build_aggregator, initialize_aggregator
export Trace, ReactionTrace, HybridTrace
export num_reactions, num_species, set_tspan
export step_ssa

using StaticArrays
import Base.show
using Setfield
using Random

abstract type AbstractJumpSet end

initialize_cache(js::AbstractJumpSet) = nothing
update_cache!(agg, js::AbstractJumpSet) = nothing


"""
    struct ReactionSet <: AbstractJumpSet

A struct that stores reactions for use with Gillespie simulation code.

# Fields
- `rates::Vector{Float64}`: Vector of reaction rates.
- `rstoich::Vector{Vector{Pair{Int64,Int64}}}`: Vector of vectors specifying the reactant stoichiometry for each reaction. Each inner vector contains pairs of species index and stoichiometric coefficient.
- `nstoich::Vector{Vector{Pair{Int64,Int64}}}`: Vector of vectors specifying the product stoichiometry for each reaction. Each inner vector contains pairs of species index and stoichiometric coefficient.
- `nspecies::Int`: Number of species in the system.

This struct represents a set of reactions to be used in Gillespie simulations. The reactions are defined by their rates, reactant stoichiometry, product stoichiometry, and the number of species in the system.
"""
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
    struct ReactionTrace <: Trace
        "a vector of reaction times"
        t::Vector{Float64}

        "a vector of reaction indices"
        rx::Vector{Int16}
    end

A (partial) record of a SSA execution.

A ReactionTrace is a data structure that captures information about the firing times and 
indices of the reactions that occurred during a SSA simulation. It consists of two fields: 
a vector of reaction firing times t and a vector of corresponding reaction indices rx.

A full reaction trace records every reaction that fired during the SSA simulation, 
whereas a partial trace only records reactions involving a specific species. This makes it 
a useful tool for studying the behavior of a particular species in a reaction system.

The `t` field stores the times at which each reaction occurred, sorted in ascending order. 
The `rx` field stores the corresponding indices of the reactions that fired, with each index 
referencing a reaction in the overall reaction system.
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

"""
    struct HybridTrace{U,T} <: Trace
        "a vector of reaction times"
        t::Vector{Float64}

        "a vector of reaction indices"
        rx::Vector{Int16}

        "a vector of external signal"
        u::U

        "sampling times of external trajectory"
        dtimes::T
    end

A hybrid trace of a discrete-time stochastic simulation with external signals.

A HybridTrace is a data structure that records the results of a hybrid simulation, which 
is a type of stochastic simulation that combines a continuous-time simulation of a reaction 
system with a discrete-time simulation of an external signal.

The HybridTrace structure contains four fields: a vector of reaction firing times t, a vector 
of corresponding reaction indices rx, a vector of external signals u, and a vector of sampling times dtimes.

The t and rx fields have the same meaning as in ReactionTrace. The u field stores the values 
of the external signal at each time step, and the dtimes field stores the times at which 
the external signal is sampled.
"""
struct HybridTrace{U,T} <: Trace
    "a vector of reaction times"
    t::Vector{Float64}

    "a vector of reaction indices"
    rx::Vector{Int16}

    "a vector of external signal"
    u::U

    "sampling times of external trajectory"
    dtimes::T
end

ReactionTrace(ht::HybridTrace) = ReactionTrace(ht.t, ht.rx)

"""
    filter_trace(trace::ReactionTrace, keep_reactions)

Filters a given `ReactionTrace` to include only the specified reaction events.

This function takes a `ReactionTrace` object and creates a new `ReactionTrace` object that only contains the reaction events specified in `keep_reactions` attribute. 
The resulting `ReactionTrace` object only includes the firing times and corresponding indices of the specified reaction events.

# Arguments
- `trace::ReactionTrace`: The original `ReactionTrace` object to be filtered.
- `keep_reactions`: The reaction events to keep in the filtered `ReactionTrace`. Can be either a single integer or an 
iterable of integers specifying the reaction indices to keep. All other reactions are discarded.

# Returns
- A new `ReactionTrace` object that includes only the specified reaction events.

# Examples
```julia
julia> trace = ReactionTrace([0.1, 0.5, 1.0, 1.5], [1, 2, 3, 2])
julia> filter_trace(trace, [2, 3])
ReactionTrace([0.5, 1.0, 1.5], [2, 3, 2])
```
"""
function filter_trace(trace::ReactionTrace, keep_reactions)
    filtered_t = Float64[]
    filtered_rx = Int16[]
    for i in 1:length(trace.t)
        if trace.rx[i] in keep_reactions
            push!(filtered_t, trace.t[i])
            push!(filtered_rx, trace.rx[i])
        end
    end
    ReactionTrace(filtered_t, filtered_rx)
end

filter_trace(trace::ReactionTrace, keep_reactions::Integer) = filter_trace(trace, (keep_reactions,))


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
struct DirectAggregator{U,Map,Cache,Rng} <: AbstractJumpRateAggregator
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

    cache::Cache

    rng::Rng
end

function build_aggregator(alg::GillespieDirect, reactions::AbstractJumpSet, ridtogroup, tspan=(0.0, Inf64); seed=rand(UInt))
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
        0.0,
        initialize_cache(reactions),
        Xoshiro(seed))
end

function Base.copy(agg::DirectAggregator)
    DirectAggregator(
        copy(agg.u),
        agg.sumrate, copy(agg.rates),
        agg.gsumrate, copy(agg.grates),
        agg.active_reactions,
        agg.traced_reactions,
        agg.ridtogroup,
        agg.tspan,
        agg.tprev,
        agg.tstop,
        agg.trace_index,
        agg.weight,
        isnothing(agg.cache) ? nothing : copy(agg.cache),
        Xoshiro(rand(agg.rng, UInt))
    )
end

struct DepGraphAggregator{U,Map,DepGraph,Cache,Rng} <: AbstractJumpRateAggregator
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

    cache::Cache

    rng::Rng
end

function build_aggregator(alg::DepGraphDirect, reactions::AbstractJumpSet, ridtogroup, tspan=(0.0, Inf64); seed=rand(UInt))
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
        collect(1:nreactions),
        initialize_cache(reactions),
        Xoshiro(seed)
    )
end

function Base.copy(agg::DepGraphAggregator)
    DepGraphAggregator(
        copy(agg.u),
        agg.sumrate, copy(agg.rates),
        agg.gsumrate, copy(agg.grates),
        agg.active_reactions,
        agg.traced_reactions,
        agg.ridtogroup,
        agg.tspan,
        agg.tprev,
        agg.tstop,
        agg.trace_index,
        agg.weight,
        agg.depgraph,
        copy(agg.jump_search_order),
        isnothing(agg.cache) ? nothing : copy(agg.cache),
        Xoshiro(rand(agg.rng, UInt))
    )
end

function initialize_aggregator(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet;
    u0=agg.u,
    tspan=(0.0, Inf64),
    active_reactions=agg.active_reactions,
    traced_reactions=agg.traced_reactions,
    seed=nothing)
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
    if seed === nothing
        Random.seed!(agg.rng)
    else
        Random.seed!(agg.rng, seed)
    end
    agg = @set agg.tstop = tspan[1] + randexp(agg.rng) / agg.sumrate
    agg
end

function initialize_aggregator(
    agg::DepGraphAggregator,
    reactions::AbstractJumpSet;
    u0=agg.u,
    tspan=(0.0, Inf64),
    active_reactions=agg.active_reactions,
    traced_reactions=agg.traced_reactions,
    seed=nothing)
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
    if seed === nothing
        Random.seed!(agg.rng)
    else
        Random.seed!(agg.rng, seed)
    end
    agg = @set agg.tstop = tspan[1] + randexp(agg.rng) / agg.sumrate
    agg = @set agg.jump_search_order = collect(active_reactions)
    agg
end

set_tspan(agg::AbstractJumpRateAggregator, tspan) = @set agg.tspan = tspan

function species_to_dependent_reaction_map(reactions::AbstractJumpSet)
    nspecies = num_species(reactions)
    # map from a species to reactions that depend on it
    spec_to_dep_rxs = [Vector{Int}() for n = 1:nspecies]
    for rx in 1:num_reactions(reactions)
        for spec in dependend_species(reactions, rx)
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

dependend_species(rs::AbstractJumpSet, index) = (spec for (spec, _) in rs.rstoich[index])
mutated_species(rs::AbstractJumpSet, index) = (spec for (spec, _) in rs.nstoich[index])
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

function step_ssa(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet,
    trace::Union{Nothing,<:Trace},
    out_trace::Union{Nothing,<:Trace};
)
    tindex = agg.trace_index

    if (!isnothing(trace)) && (tindex <= length(trace.t)) && (agg.tstop >= trace.t[tindex])
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
        new_tstop = tnow + randexp(agg.rng) / agg.sumrate
        agg = @set agg.tstop = new_tstop
    end

    # store reaction event in trace
    if (!isnothing(out_trace)) && rx ∈ agg.traced_reactions
        push!(out_trace.rx, rx)
        push!(out_trace.t, tnow)
    end

    agg
end

@inline function select_reaction(agg::DirectAggregator)
    rx = 0
    x = rand(agg.rng) * agg.sumrate
    @inbounds for reaction in agg.active_reactions
        rx = reaction
        if x <= agg.rates[rx]
            break
        end
        x -= agg.rates[rx]
    end
    rx
end

@inline function select_reaction(agg::DepGraphAggregator)
    rx = 0
    jsidx = 1
    x = rand(agg.rng) * agg.sumrate
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

@inline function update_weight(agg::AbstractJumpRateAggregator, tnow, rx=nothing)
    # compute log probability of jump; update weight
    Δt = tnow - agg.tprev
    log_jump_prob = 0.0
    if !isnothing(rx)
        @inbounds gid = agg.ridtogroup[rx]
        gid != 0 && @inbounds log_jump_prob += log(agg.grates[gid])
    end
    log_waiting_prob = -Δt * agg.gsumrate
    agg = @set agg.weight = agg.weight + log_jump_prob + log_waiting_prob
    agg = @set agg.tprev = tnow
end

# Code adapted from JumpProcesses.jl
@inline function executerx!(speciesvec::AbstractVector, rxidx::Integer, reactions::AbstractJumpSet)
    @inbounds net_stoch = reactions.nstoich[rxidx]
    for (species, diff) in net_stoch
        @inbounds speciesvec[species] += diff
    end
    nothing
end

@inline function executerx!(speciesvec::AbstractVector, rxidx::Integer, js::JumpSet)
    nreactions = num_reactions(js.reactions)
    if rxidx <= nreactions
        executerx!(speciesvec, rxidx, js.reactions)
    else
        executerx!(speciesvec, rxidx - nreactions, js.jumps)
    end
end

@inline function evalrxrate(speciesvec::AbstractVector, rxidx::Int64, rs::ReactionSet)
    @inbounds rstoich = rs.rstoich[rxidx]
    @inbounds rate = rs.rates[rxidx]
    rate_mult = one(eltype(speciesvec))
    for (species, count) in rstoich
        @inbounds specpop = speciesvec[species]
        rate_mult *= specpop
        for k = 2:count
            specpop -= 1
            rate_mult *= specpop
        end
    end
    rate * rate_mult
end

@inline function evalrxrate(agg::AbstractJumpRateAggregator, rxidx::Int64, rs::ReactionSet)
    evalrxrate(agg.u, rxidx, rs)
end

@inline function evalrxrate(agg::AbstractJumpRateAggregator, rxidx::Int64, js::JumpSet)
    speciesvec = agg.u
    nreactions = num_reactions(js.reactions)
    if rxidx <= nreactions
        evalrxrate(speciesvec, rxidx, js.reactions)
    else
        evalrxrate(speciesvec, rxidx - nreactions, js.jumps)
    end
end

@inline function update_reaction_rate!(aggregator::AbstractJumpRateAggregator, reactions::AbstractJumpSet, rx::Int)
    rate = evalrxrate(aggregator, rx, reactions)
    @inbounds gid = aggregator.ridtogroup[rx]
    if gid != 0
        @inbounds aggregator.grates[gid] += -aggregator.rates[rx] + rate
    end
    @inbounds aggregator.rates[rx] = rate
    rate
end

function update_rates(aggregator::DirectAggregator, reactions::AbstractJumpSet, prev_reaction::Integer=0)
    update_cache!(aggregator, reactions)
    for rx in 1:num_reactions(reactions)
        update_reaction_rate!(aggregator, reactions, rx)
    end

    sumrate = sum(rx -> (@inbounds aggregator.rates[rx]), aggregator.active_reactions; init=zero(eltype(aggregator.rates)))

    aggregator = @set aggregator.gsumrate = sum(aggregator.grates)
    aggregator = @set aggregator.sumrate = sumrate
end

function update_rates(aggregator::DepGraphAggregator, reactions::AbstractJumpSet, prev_reaction::Integer=0)
    update_cache!(aggregator, reactions)
    if prev_reaction == 0
        # recompute all rates
        for rx in 1:num_reactions(reactions)
            update_reaction_rate!(aggregator, reactions, rx)
        end
        sumrate = sum(rx -> aggregator.rates[rx], aggregator.active_reactions)
    else
        # only recompute dependent rates
        sumrate = aggregator.sumrate + sum(aggregator.depgraph[prev_reaction]; init=zero(eltype(aggregator.rates))) do rx
            @inbounds oldrate = aggregator.rates[rx]
            newrate = update_reaction_rate!(aggregator, reactions, rx)
            rx ∈ aggregator.active_reactions ? (newrate - oldrate) : zero(newrate)
        end
    end

    aggregator = @set aggregator.gsumrate = sum(aggregator.grates)
    aggregator = @set aggregator.sumrate = sumrate
end

end # module
