import ModelingToolkit
import ModelingToolkit: build_function, substitute
import Catalyst
import Catalyst: ReactionSystem
using StaticArrays
import Base.show
using Setfield

struct ReactionSet
    rates::Vector{Float64}
    rstoich::Vector{Vector{Pair{Int64,Int64}}}
    nstoich::Vector{Vector{Pair{Int64,Int64}}}
    nspecies::Int
end

abstract type AbstractJumpRateAggregatorAlgorithm end

struct GillespieDirect <: AbstractJumpRateAggregatorAlgorithm end

struct DepGraphDirect <: AbstractJumpRateAggregatorAlgorithm end

abstract type AbstractJumpRateAggregator end

struct DirectAggregator{U} <: AbstractJumpRateAggregator
    "the sum of all reaction rates"
    sumrate::Float64

    "a vector of the current reaction rates"
    rates::Vector{Float64}

    "sum of group propensities"
    gsums::Vector{Float64}

    "maps reaction indices to group indices"
    ridtogroup::U

    "time span for aggregation"
    tspan::Tuple{Float64,Float64}

    "time of last recorded reaction"
    tprev::Float64

    "accumulated log-probability"
    weight::Float64
end

function build_aggregator(alg::GillespieDirect, reactions::ReactionSet, ridtogroup, tspan=(0.0, Inf64))
    ngroups = maximum(ridtogroup)
    nreactions = length(reactions.rates)
    DirectAggregator(0.0, zeros(nreactions), zeros(ngroups), ridtogroup, tspan, tspan[1], 0.0)
end

struct DepGraphAggregator{U,DepGraph} <: AbstractJumpRateAggregator
    "the sum of all reaction rates"
    sumrate::Float64

    "a vector of the current reaction rates"
    rates::Vector{Float64}

    "sum of group propensities"
    gsums::Vector{Float64}

    "maps reaction indices to group indices"
    ridtogroup::U

    "time span for aggregation"
    tspan::Tuple{Float64,Float64}

    "time of last recorded reaction"
    tprev::Float64

    "accumulated log-probability"
    weight::Float64

    "dependency graph"
    depgraph::DepGraph

    "previously executed reaction"
    prev_reaction::Int
end

function build_aggregator(alg::DepGraphDirect, reactions::ReactionSet, ridtogroup, tspan=(0.0, Inf64))
    ngroups = maximum(ridtogroup)
    nreactions = length(reactions.rates)
    depgraph = make_depgraph(reactions)
    DepGraphAggregator(0.0, zeros(nreactions), zeros(ngroups), ridtogroup, tspan, tspan[1], 0.0, depgraph, 0)
end

function initialize_aggregator(agg::AbstractJumpRateAggregator; tspan=(0.0, Inf64))
    agg = @set agg.tspan = tspan
    agg = @set agg.tprev = tspan[1]
    agg = @set agg.weight = 0.0
    agg = @set agg.sumrate = 0.0
    agg.rates .= 0.0
    agg.gsums .= 0.0
    agg
end

function initialize_aggregator(agg::DepGraphAggregator; tspan=(0.0, Inf64))
    agg = @set agg.tspan = tspan
    agg = @set agg.tprev = tspan[1]
    agg = @set agg.weight = 0.0
    agg = @set agg.sumrate = 0.0
    agg = @set agg.prev_reaction = 0
    agg.rates .= 0.0
    agg.gsums .= 0.0
    agg
end

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

function species_to_dependent_reaction_map(reactions::ReactionSet)
    nspecies = reactions.nspecies
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

function make_depgraph(reactions::ReactionSet)
    nreactions = length(reactions.rates)
    spec_to_dep_rxs = species_to_dependent_reaction_map(reactions)

    # create map from rx to reactions depending on it
    dep_graph = [Vector{Int}() for n = 1:nreactions]
    for rx in 1:nreactions
        # rx changes spec, hence rxs depending on spec depend on rx
        for (spec, stoch) in reactions.nstoich[rx]
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
        agg = update_rates(agg, (u, t, i), dist.reactions)

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
    agg = update_rates(agg, (u, t, i), dist.reactions)
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
        agg = update_rates(agg, (u, t, i), dist.reactions)

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

@inline function update_reaction_rate!(aggregator::AbstractJumpRateAggregator, u::AbstractVector, reactions, rx::Int)
    @inbounds gid = aggregator.ridtogroup[rx]
    if gid != 0
        rate = evalrxrate(u, rx, reactions)
        @inbounds aggregator.gsums[gid] += -aggregator.rates[rx] + rate
        @inbounds aggregator.rates[rx] = rate
    end
end

@fastmath function update_rates(aggregator::DirectAggregator, (u, t, i), reactions::ReactionSet)
    for rx in eachindex(reactions.rates)
        update_reaction_rate!(aggregator, u, reactions, rx)
    end
    aggregator = @set aggregator.sumrate = sum(aggregator.gsums)
end

@fastmath function update_rates(aggregator::DepGraphAggregator, (u, t, i), reactions::ReactionSet)
    if aggregator.prev_reaction == 0
        for rx in eachindex(reactions.rates)
            update_reaction_rate!(aggregator, u, reactions, rx)
        end
    else
        @inbounds for rx in aggregator.depgraph[aggregator.prev_reaction]
            update_reaction_rate!(aggregator, u, reactions, rx)
        end
    end
    aggregator = @set aggregator.sumrate = sum(aggregator.gsums)
    aggregator = @set aggregator.prev_reaction = i
end
