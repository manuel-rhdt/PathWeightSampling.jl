using ModelingToolkit

# needed to work around issue in ModelingToolkit where the call to jumpratelaw fails
# if the rate is a simple number instead of an operation
ModelingToolkit.get_variables(f::Number) = Variable[]

function build_rate_functions(reaction_network::ReactionSystem)
    totalrate = sum((jumpratelaw(react) for react in reactions(reaction_network)))
    totalrate_fun = build_function(totalrate, species(reaction_network), params(reaction_network); expression=Val{false})

    symbol_map = speciesmap(reaction_network)
    rates = Dict{Vector{Int64},ModelingToolkit.Operation}()
    for react in reactions(reaction_network)
        net_change = zeros(Int64, numspecies(reaction_network))
        for (species, change) in react.netstoich
            net_change[symbol_map[species]] = change
        end

        rate = jumpratelaw(react)
        
        if haskey(rates, net_change)
            rates[net_change] += rate
        else
            rates[net_change] = rate
        end
    end

    rate_funs = [
        change => build_function(expr, species(reaction_network), params(reaction_network); expression=Val{false}) for (change, expr) in pairs(rates)
    ]
    (totalrate_fun, rate_funs)
end

struct TrajectoryDistribution{TRate}
    totalrate::TRate
    rates::Array{Pair{Vector{Int64}, LogRate} where LogRate}
end

distribution(rn::ReactionSystem) = TrajectoryDistribution(build_rate_functions(rn)...)

function logpdf(dist::TrajectoryDistribution, trajectory; params=[])
    result = 0.0
    tprev = nothing
    uprev = nothing
    du = nothing

    for (u, t) in trajectory
        if tprev === nothing
            tprev = t
            uprev = copy(u)
            du = u - uprev
            continue
        end

        dt = t - tprev
        du .= u .- uprev

        result += - dt * dist.totalrate(uprev, params)

        for (net_change, rate_fun) in dist.rates
            if net_change == du
                result += log(rate_fun(uprev, params))
                break
            end
        end

        tprev = t
        uprev[:] .= u
    end

    result
end