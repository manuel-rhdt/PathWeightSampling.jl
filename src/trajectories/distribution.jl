using ModelingToolkit

struct ChemicalReaction
    rate::Float64
    substoich::Vector{Tuple{Int,Int}}
    netstoich::Vector{Tuple{Int,Int}}
end

struct TrajectoryDistribution
    reactions::Vector{ChemicalReaction}
end

distribution(rn::ReactionSystem) = TrajectoryDistribution(create_chemical_reactions(rn))

@fastmath function logpdf(dist::TrajectoryDistribution, trajectory; params=[])
    result = 0.0
    ((uprev, tprev), state) = iterate(trajectory)

    for (u, t) in Iterators.rest(trajectory, state)
        dt = t - tprev
        du = u - uprev

        result += - dt * evaltotalrate(uprev, dist.reactions)

        if !allzero(du)
            idx = findreactions(du, dist.reactions)
            if idx > 0
                @inbounds rate = evalrxrate(uprev, dist.reactions[idx])
                result += log(rate)
            end
        end

        tprev = t
        uprev = copy(u)
    end

    result
end

function create_chemical_reactions(reaction_system::ReactionSystem)
    smap = speciesmap(reaction_system)
    map(reactions(reaction_system)) do (mreact)
        rate = mreact.rate
        substoich = sort([(smap[sub.op], stoich) for (sub, stoich) in zip(mreact.substrates, mreact.substoich)])
        netstoich = sort([(smap[sub], stoich) for (sub, stoich) in mreact.netstoich])
        ChemicalReaction(rate, substoich, netstoich)
    end
end

@inline function allzero(array::AbstractArray)
    for i in eachindex(array)
        @inbounds if array[i] != 0
            return false
        end
    end
    true
end

@inline function allzero(array::AbstractArray, range::AbstractRange)
    for i in eachindex(range)
        @inbounds if array[i] != 0
            return false
        end
    end
    true
end

@inline @fastmath function findreactions(du::AbstractVector, reactions::Vector{ChemicalReaction})::Int
    for (j, reaction) in enumerate(reactions)
        i = 1
        found = true
        for (index, netstoich) in reaction.netstoich
            if !allzero(du, i:index - 1)
                found = false
                break
            else
                i = index + 1
                @inbounds if du[index] != netstoich
                    found = false
                    break
                end
            end
        end
        if found
            return j
        end
    end
    0
end

# taken from DiffEqJump
@inline @fastmath function evalrxrate(speciesvec::AbstractVector, reaction::ChemicalReaction)
    val = 1.0
    for specstoch in reaction.substoich
        @inbounds specpop = speciesvec[specstoch[1]]
        val    *= specpop
        @inbounds for k = 2:specstoch[2]
            specpop -= one(specpop)
            val     *= specpop
        end
    end
    val * reaction.rate
end

evaltotalrate(speciesvec::AbstractVector, reactions::Vector{ChemicalReaction}) = sum(r -> evalrxrate(speciesvec, r), reactions)
