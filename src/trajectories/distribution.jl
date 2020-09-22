using ModelingToolkit

# needed to work around issue in ModelingToolkit where the call to jumpratelaw fails
# if the rate is a simple number instead of an operation
ModelingToolkit.get_variables(f::Number) = Variable[]

function build_rate_functions(reaction_network::ReactionSystem)
    totalrate = sum((jumpratelaw(react) for react in reactions(reaction_network)))
    totalrate_fun = build_function(totalrate, species(reaction_network), params(reaction_network); expression=Val{false})

    rate_funs = build_reaction_rate(reaction_network, reactions(reaction_network)...)
    # symbol_map = speciesmap(reaction_network)
    # rates = Dict{Vector{Int64},ModelingToolkit.Operation}()
    # for react in reactions(reaction_network)
    #     net_change = zeros(Int64, numspecies(reaction_network))
    #     for (species, change) in react.netstoich
    #         net_change[symbol_map[species]] = change
    #     end

    #     rate = jumpratelaw(react)
        
    #     if haskey(rates, net_change)
    #         rates[net_change] += rate
    #     else
    #         rates[net_change] = rate
    #     end
    # end

    # rate_funs = [
    #     change => build_function(expr, species(reaction_network), params(reaction_network); expression=Val{false}) for (change, expr) in pairs(rates)
    # ]
    (totalrate_fun, rate_funs)
end

function build_rate_function_derivatives(reaction_network::ReactionSystem)
    totalrate::Operation = sum((jumpratelaw(react) for react in reactions(reaction_network)))

    interaction_param = params(reaction_network)[1]()
    @derivatives D'~interaction_param
    d_totalrate = expand_derivatives(D(totalrate))
    totalrate_fun = build_function(d_totalrate, species(reaction_network), params(reaction_network); expression=Val{false})

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

    for key in keys(rates)
        rates[key] = expand_derivatives(D(log(rates[key])))
    end

    println(rates)
    rate_funs = [
        change => build_function(expr, species(reaction_network), params(reaction_network); expression=Val{false}) for (change, expr) in pairs(rates)
    ]
    (totalrate_fun, rate_funs)
end

struct ReactionRate{Rate}
    net_change::Vector{Int}
    rate::Rate
end

function build_reaction_rate(network::ReactionSystem, reaction::Reaction)
    symbol_map = speciesmap(network)

    net_change = zeros(Int, numspecies(network))
    for (species, change) in reaction.netstoich
        net_change[symbol_map[species]] = change
    end
    expr = jumpratelaw(reaction)
    rate = build_function(expr, species(network), params(network); expression=Val{false})

    ReactionRate(net_change, rate)
end

build_reaction_rate(network::ReactionSystem, r1::Reaction, r2::Reaction) = (build_reaction_rate(network, r1), build_reaction_rate(network, r2))

build_reaction_rate(network::ReactionSystem, r1::Reaction, r2::Reaction, rother::Reaction...) = (build_reaction_rate(network, r1), build_reaction_rate(network, r2, rother...)...)

function evaluate_rate(acc, du, uprev, params, rate::ReactionRate)
    if du == rate.net_change
        return acc + log(rate.rate(uprev, params))
    else
        return acc
    end
end

function evaluate_rate(acc, du, uprev, params, rate::ReactionRate, rates::ReactionRate...)
    new_acc = evaluate_rate(acc, du, uprev, params, rate)
    evaluate_rate(new_acc, du, uprev, params, rates...)
end

struct TrajectoryDistribution{TRate,Rates}
    totalrate::TRate
    rates::Rates
end

distribution(rn::ReactionSystem) = TrajectoryDistribution(build_rate_functions(rn)...)

function logpdf(dist::TrajectoryDistribution, trajectory; params=[])
    result = 0.0
    ((uprev, tprev), state) = iterate(trajectory)

    for (u, t) in Iterators.rest(trajectory, state)
        dt = t - tprev
        du = u - uprev

        result += - dt * dist.totalrate(uprev, params)
        result += evaluate_rate(0.0, du, uprev, params, dist.rates...)

        tprev = t
        uprev = u
    end

    result
end