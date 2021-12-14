import ModelingToolkit
import ModelingToolkit: build_function, substitute
import Catalyst
import Catalyst: ReactionSystem
using StaticArrays
using Transducers
import Base.show

struct DirectAggregator{U}
    sumrate::Float64
    rates::Vector{Float64}
    # a mapping from the trajectory's recorded reaction ids to the ReactionSet indices of the jump aggregator
    update_map::U
    tspan::Tuple{Float64, Float64}
    tprev::Float64
    weight::Float64
end

struct ReactionSet
    rates::Vector{Float64}
    rstoich::Vector{Vector{Pair{Int64, Int64}}}
    nstoich::Vector{Vector{Pair{Int64, Int64}}}
end

function ReactionSet(js::ModelingToolkit.JumpSystem, p)
    parammap  = map(Pair, ModelingToolkit.parameters(js), p)
    statetoid = Dict(ModelingToolkit.value(state) => i for (i,state) in enumerate(ModelingToolkit.states(js)))

    rates = Float64[]
    rstoich_vec = Vector{Pair{Int64, Int64}}[]
    nstoich_vec = Vector{Pair{Int64, Int64}}[]

    for eq in ModelingToolkit.equations(js)
        rate = ModelingToolkit.value(ModelingToolkit.substitute(eq.scaled_rates, parammap))
        rstoich = sort!([statetoid[ModelingToolkit.value(spec)] => stoich for (spec, stoich) in eq.reactant_stoch])
        nstoich = sort!([statetoid[ModelingToolkit.value(spec)] => stoich for (spec, stoich) in eq.net_stoch])

        push!(rates, rate)
        push!(rstoich_vec, rstoich)
        push!(nstoich_vec, nstoich)
    end

    ReactionSet(rates, rstoich_vec, nstoich_vec)
end

add_weight(agg::DirectAggregator, Δweight::Float64, t::Float64) = DirectAggregator(agg.sumrate, agg.rates, agg.update_map, agg.tspan, t, agg.weight + Δweight)

function get_update_index(agg::DirectAggregator, i::Int) 
    if checkbounds(Bool, agg.update_map, i) 
        j = @inbounds agg.update_map[i] 
        if j > 0
            j
        else
            nothing
        end
    else
        nothing
    end
end

struct TrajectoryDistribution{U}
    reactions::ReactionSet
    aggregator::DirectAggregator{U}
end

function TrajectoryDistribution(reactions, update_map = 1:num_reactions)
    num_clusters = maximum(update_map)
    agg = DirectAggregator(0.0, zeros(num_clusters), update_map, (0.0, 0.0), 0.0, 0.0)
    TrajectoryDistribution(reactions, agg)
end

distribution(rn::ReactionSystem, p; update_map=1:Catalyst.numreactions(rn)) = TrajectoryDistribution(ReactionSet(convert(ModelingToolkit.JumpSystem, rn), p), update_map)

@fastmath function Distributions.logpdf(dist::TrajectoryDistribution, trajectory)::Float64
    first = iterate(trajectory)
    if first === nothing
        return 0.0
    end
    tprev = 0.0
    result = 0.0
    
    agg = dist.aggregator
    for (u, t, i) in trajectory
        agg = update_rates(agg, u, dist.reactions)

        dt = t - tprev
        result -= dt * agg.sumrate
        agg_i = get_update_index(agg, i)
        if agg_i !== nothing
            @inbounds result += log(agg.rates[agg_i])
        end
    
        tprev = t
    end

    result
end

@fastmath @inline function fold_logpdf(dist::TrajectoryDistribution, agg::DirectAggregator, (u, t, i))
    agg = update_rates(agg, u, dist.reactions)
    agg_i = get_update_index(agg, i)
    dt = t - agg.tprev
    if agg_i !== nothing
        @inbounds log_jump_prob = log(agg.rates[agg_i])
    else
        log_jump_prob = 0.0
    end
    log_surv_prob = - dt * agg.sumrate
    add_weight(agg, log_surv_prob + log_jump_prob, t)
end

function trajectory_energy(dist::TrajectoryDistribution, traj; tspan=(0.0, Inf64))
    agg = dist.aggregator
    agg = DirectAggregator(0.0, agg.rates, agg.update_map, tspan, tspan[1], 0.0)
    
    f = let dist=dist 
        function(agg, (u, t, i))
            if t <= agg.tspan[1]
                return agg
            end
            if t > agg.tspan[2]
                agg = fold_logpdf(dist, agg, (u, agg.tspan[2], 0))
                return Transducers.reduced(agg)
            else
                return fold_logpdf(dist, agg, (u, t, i))
            end
        end 
    end
    
    result = foldxl(f, traj; init=agg) 
    result.weight
end

function cumulative_logpdf!(result::AbstractVector, dist::TrajectoryDistribution, traj, dtimes::AbstractVector)
    agg = dist.aggregator
    tspan = (first(dtimes), last(dtimes))
    result[1] = zero(eltype(result))
    agg = DirectAggregator(0.0, agg.rates, agg.update_map, tspan, tspan[1], 0.0)
    foldxl(traj; init=(agg, 1)) do (agg, k), (u, t, i)
        if t <= agg.tspan[1]
            return agg, k
        end

        t = min(t, agg.tspan[2])
        agg = update_rates(agg, u, dist.reactions)

        tprev = agg.tprev
        while k <= length(dtimes) && dtimes[k] < t
            result[k] -= (dtimes[k] - tprev) * agg.sumrate
            tprev = dtimes[k]
            k += 1
            result[k] = result[k - 1]
        end
        result[k] -= (t - tprev) * agg.sumrate

        agg_i = get_update_index(agg, i)
        if agg_i !== nothing
            @inbounds log_jump_prob = log(agg.rates[agg_i])
        else
            log_jump_prob = 0.0
        end
        result[k] += log_jump_prob
       
        agg = add_weight(agg, -(t - agg.tprev) * agg.sumrate + log_jump_prob, t)

        if t >= agg.tspan[2]
            return Transducers.reduced((agg, k))
        else
            return agg, k
        end
    end
    
    result
end

cumulative_logpdf(dist::TrajectoryDistribution, trajectory, dtimes::AbstractVector) = cumulative_logpdf!(zeros(length(dtimes)), dist, trajectory, dtimes)


@inline @fastmath function evalrxrate(speciesvec::AbstractVector, rxidx::Int64, rs::ReactionSet)
    val = 1.0
    @inbounds for specstoch in rs.rstoich[rxidx]
        specpop = speciesvec[specstoch[1]]
        val    *= specpop
        @inbounds for k = 2:specstoch[2]
            specpop -= one(specpop)
            val     *= specpop
        end
    end

    @inbounds return val * rs.rates[rxidx]
end

@fastmath function update_rates(aggregator::DirectAggregator, speciesvec::AbstractVector, reactions::ReactionSet)
    sum = 0.0
    aggregator.rates .= 0.0
    for i in eachindex(reactions.rates)
        agg_i = get_update_index(aggregator, i)
        if agg_i !== nothing
            rate = evalrxrate(speciesvec, i, reactions)
            @inbounds aggregator.rates[agg_i] += rate
            sum += rate
        end
    end
    DirectAggregator(sum, aggregator.rates, aggregator.update_map, aggregator.tspan, aggregator.tprev, aggregator.weight)
end
