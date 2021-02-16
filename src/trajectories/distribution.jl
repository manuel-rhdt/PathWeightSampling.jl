import ModelingToolkit
import ModelingToolkit: build_function, ReactionSystem, substitute
import Catalyst

struct ChemicalReaction{Rate,N}
    rate::Rate
    netstoich::SVector{N,Int}
end

struct TrajectoryDistribution{Reactions,Dist}
    reactions::Reactions
    log_p0::Dist
end

distribution(rn::ReactionSystem) = distribution(rn, (x...) -> 0.0)
distribution(rn::ReactionSystem, log_p0) = TrajectoryDistribution(create_chemical_reactions(rn), log_p0)

@fastmath function Distributions.logpdf(dist::TrajectoryDistribution{<:Tuple}, trajectory; params=[])::Float64
    first = iterate(trajectory)
    if first === nothing
        return 0.0
    end
    ((uprev, tprev), state) = first
    result = dist.log_p0(uprev...)::Float64
    
    for (u, t) in Iterators.rest(trajectory, state)
        dt = t - tprev
        du = u - uprev

        totalrate = evaltotalrate(uprev, dist.reactions..., params=params)

        result -= dt * totalrate
        reaction_rate = evalrxrate(uprev, du, dist.reactions..., params=params)
        if reaction_rate != zero(reaction_rate)
            result += log(reaction_rate)
        end

        tprev = t
        uprev = copy(u)
    end

    result
end

function cumulative_logpdf!(result::AbstractVector, dist::TrajectoryDistribution{<:Tuple}, trajectory, times::AbstractVector; params=[])
    ((uprev, tprev), state) = iterate(trajectory)
    
    j = 1
    while j <= length(times) && tprev > times[j]
        result[j] = 0.0
        j += 1
    end
    result[j] = dist.log_p0(uprev...)::Float64

    for (u, t) in Iterators.rest(trajectory, state)
        du = u - uprev

        totalrate = evaltotalrate(uprev, dist.reactions..., params=params)

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

        reaction_rate = evalrxrate(uprev, du, dist.reactions..., params=params)
        log_reaction_rate = reaction_rate == 0 ? zero(reaction_rate) : log(reaction_rate)
        result[j] += - dt * totalrate + log_reaction_rate

        tprev = t
        uprev = copy(u)
    end

    if length(result) > j
        result[j + 1:end] .= result[j]
    end

    result
end

cumulative_logpdf(dist::TrajectoryDistribution{<:Tuple}, trajectory, dtimes::AbstractVector; params=[]) = cumulative_logpdf!(empty(dtimes, Float64), dist, trajectory, times, params=params)

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

@inline @fastmath function evalrxrate(speciesvec::AbstractVector, reaction::ChemicalReaction, params=[])::Float64
    (reaction.rate)(speciesvec, params)
end

@inline function evalrxrate(speciesvec::AbstractVector, du::AbstractVector; params=[])::Float64
    0.0
end

@inline @fastmath function evalrxrate(speciesvec::AbstractVector, du::AbstractVector, r1::ChemicalReaction, rs::ChemicalReaction...; params=[])::Float64
    if du == r1.netstoich
        return evalrxrate(speciesvec, r1, params) + evalrxrate(speciesvec, du, rs..., params=params)
    else
        return evalrxrate(speciesvec, du, rs..., params=params)
    end
end

@inline @fastmath function evaltotalrate(speciesvec::AbstractVector, r1::ChemicalReaction; params=[])::Float64
    evalrxrate(speciesvec, r1, params)
end

@inline @fastmath function evaltotalrate(speciesvec::AbstractVector, r1::ChemicalReaction, rs::ChemicalReaction...; params=[])::Float64
    evalrxrate(speciesvec, r1, params) + evaltotalrate(speciesvec, rs..., params=params)
end

