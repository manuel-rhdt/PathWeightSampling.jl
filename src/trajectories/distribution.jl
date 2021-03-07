import ModelingToolkit
import ModelingToolkit: build_function, ReactionSystem, substitute
import Catalyst

mutable struct DirectAggregator{U}
    sumrate::Float64
    rates::Vector{Float64}
    update_map::U
end

get_update_index(agg::DirectAggregator, i::Int) = checkbounds(Bool, agg.update_map, i) ? begin @inbounds agg.update_map[i] end : 0

struct ChemicalReaction{Rate,N}
    rate::Rate
    netstoich::SVector{N,Int}
end

struct TrajectoryDistribution{Reactions,Dist,U}
    reactions::Reactions
    log_p0::Dist
    aggregator::DirectAggregator{U}
end

function TrajectoryDistribution(reactions, log_p0, update_map = 1:num_reactions)
    num_clusters = maximum(update_map)
    agg = DirectAggregator(0.0, zeros(num_clusters), update_map)
    TrajectoryDistribution(reactions, log_p0, agg)
end

myzero_fn(x...) = 0.0
distribution(rn::ReactionSystem, log_p0=myzero_fn; update_map=1:Catalyst.numreactions(rn)) = TrajectoryDistribution(create_chemical_reactions(rn), log_p0, update_map)

@fastmath function Distributions.logpdf(dist::TrajectoryDistribution{<:Tuple}, trajectory; params=[])::Float64
    first = iterate(trajectory)
    if first === nothing
        return 0.0
    end
    ((uprev, tprev, iprev), state) = first
    result = dist.log_p0(uprev...)::Float64
    update_rates!(dist.aggregator, uprev, params, dist.reactions...)
    
    for (u, t, i) in Iterators.rest(trajectory, state)
        dt = t - tprev

        result -= dt * dist.aggregator.sumrate
        agg_i = get_update_index(dist.aggregator, i)
        if agg_i != 0
            result += log(dist.aggregator.rates[agg_i])
        end
    
        update_rates!(dist.aggregator, u, params, dist.reactions...)

        tprev = t
    end

    result
end

function cumulative_logpdf!(result::AbstractVector, dist::TrajectoryDistribution{<:Tuple}, trajectory, times::AbstractVector; params=[])
    ((uprev, tprev, iprev), state) = iterate(trajectory)
    
    j = 1
    while j <= length(times) && tprev > times[j]
        result[j] = 0.0
        j += 1
    end
    result[j] = dist.log_p0(uprev...)::Float64
    update_rates!(dist.aggregator, uprev, params, dist.reactions...)

    for (u, t, i) in Iterators.rest(trajectory, state)
        totalrate = dist.aggregator.sumrate

        while j <= length(times) && times[j] < t
            result[j] -= (times[j] - tprev) * totalrate
            tprev = times[j]
            j += 1
            if j <= length(times)
                @inbounds result[j] = result[j - 1]
            end
        end

        if j > length(times)
            break
        end    
        
        dt = t - tprev

        agg_i = get_update_index(dist.aggregator, i)
        if agg_i != 0
            result[j] += log(dist.aggregator.rates[agg_i])
        end

        result[j] -= dt * totalrate

        update_rates!(dist.aggregator, u, params, dist.reactions...)

        tprev = t
    end

    if length(result) > j
        result[j + 1:end] .= result[j]
    end

    result
end

cumulative_logpdf(dist::TrajectoryDistribution{<:Tuple}, trajectory, dtimes::AbstractVector; params=[]) = cumulative_logpdf!(zeros(length(dtimes)), dist, trajectory, dtimes, params=params)

function create_chemical_reactions(reaction_system::ReactionSystem)
    _create_chemical_reactions(reaction_system, Catalyst.reactions(reaction_system)...)
end

function var2name(var)
    ModelingToolkit.operation(var).name
end

function _create_chemical_reactions(rn::ReactionSystem, r1::Catalyst.Reaction)
    smap = Catalyst.speciesmap(rn)
    spec = var2name.(Catalyst.species(rn))
    ratelaw = substitute(Catalyst.jumpratelaw(r1), Dict(Catalyst.species(rn) .=> spec))

    rate_fun = build_function(ratelaw, spec, Catalyst.params(rn))
    rate_fun = eval(rate_fun)

    netstoich = [(smap[sub], stoich) for (sub, stoich) in r1.netstoich]

    du = zero(SVector{Catalyst.numspecies(rn),Int})
    for (index, netstoich) in netstoich
        du = setindex(du, netstoich, index)
    end

    (ChemicalReaction(rate_fun, du),)
end

function _create_chemical_reactions(rn::ReactionSystem, r1::Catalyst.Reaction, rs::Catalyst.Reaction...)
    (_create_chemical_reactions(rn, r1)..., _create_chemical_reactions(rn, rs...)...)
end

@inline function reaction_index(du, rs::ChemicalReaction...)
    reaction_index(du, 1, rs...)
end

@inline function reaction_index(du, i::Int, r1::ChemicalReaction)
    if du == r1.netstoich
        i
    else
        0
    end
end

@inline function reaction_index(du, i::Int, r1::ChemicalReaction, rs::ChemicalReaction...)
    if du == r1.netstoich
        i
    else
        reaction_index(du, i+1, rs...)
    end
end


@inline @fastmath function update_rates!(aggregator::DirectAggregator, speciesvec::AbstractVector, params::AbstractVector{Float64}, rs::ChemicalReaction...)
    aggregator.sumrate = 0.0
    update_rates!(aggregator, 1, speciesvec, params, rs...)
end

@inline @fastmath function evalrxrate(speciesvec::AbstractVector, reaction::ChemicalReaction, params=[])::Float64
    (reaction.rate)(speciesvec, params)
end

@inline @fastmath function update_rates!(aggregator::DirectAggregator, i::Int, speciesvec::AbstractVector, params::AbstractVector{Float64}, r1::ChemicalReaction)
    agg_i = get_update_index(aggregator, i)
    if agg_i != 0
        rate = evalrxrate(speciesvec, r1, params)
        aggregator.sumrate += rate
        @inbounds aggregator.rates[agg_i] = rate
    end
    nothing
end

@inline @fastmath function update_rates!(aggregator::DirectAggregator, i::Int, speciesvec::AbstractVector, params::AbstractVector{Float64}, r1::ChemicalReaction, rs::ChemicalReaction...)
    update_rates!(aggregator, i, speciesvec, params, r1)
    update_rates!(aggregator, i+1, speciesvec, params, rs...)
end

