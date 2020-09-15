using Plots
using DifferentialEquations
using Catalyst
using ModelingToolkit


signal_network = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

response_network = @reaction_network begin
    0.01 * θ, S --> X + S
    0.01 * 50.0 * (1.0 - θ), ∅ --> X
    0.01, X --> ∅ 
end θ

# joint_network = merge(signal_network, response_network)

joint_network = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
    0.01 * θ, S --> X + S
    0.01 * 50.0 * (1.0 - θ), ∅ --> X
    0.01, X --> ∅ 
end θ

@variables p
addspecies!(joint_network, p)

u0 = [50., 50., 0.0]
p = [0.0]
tspan = (0., 500.)

mutable struct ProbabilityAggregator{T}
    previous_jump_rate::T
    previous_sum_rate::T
    delta_t::T
end

ProbabilityAggregator{T}() where T = ProbabilityAggregator(zero(T), zero(T), zero(T))

function (p::ProbabilityAggregator)(callback, u, t, integrator) # initialize!
    aggregation = integrator.cb.condition
    p.delta_t = aggregation.next_jump_time - integrator.t
    p.previous_sum_rate = aggregation.sum_rate
    p.previous_jump_rate = one(typeof(p.previous_jump_rate))
end

function (p::ProbabilityAggregator)(integrator)
    integrator.u[3] += log(p.previous_jump_rate) - p.delta_t * p.previous_sum_rate

    aggregation = integrator.cb.condition
    p.delta_t = aggregation.next_jump_time - integrator.t
    p.previous_sum_rate = aggregation.sum_rate
    nj = aggregation.next_jump
    if nj == 1
        p.previous_jump_rate = aggregation.cur_rates[1]
    else
        p.previous_jump_rate = aggregation.cur_rates[nj] - aggregation.cur_rates[nj - 1]
    end
end

pcb = ProbabilityAggregator{Float64}()

cb = DiscreteCallback((u, t, integrator) -> true, pcb; initialize=pcb, save_positions=(false, false))

dproblem = DiscreteProblem(joint_network, u0, tspan, p)
jump_problem = JumpProblem(joint_network, dproblem, Direct(), save_positions=(false, false))

solution = solve(jump_problem, SSAStepper(), callback=cb, saveat=1)

plot(solution)

solution.u

function signal_response_pair(dproblem; θ=1.0)
    dproblem = remake(dproblem; p=[θ])
    jump_problem = JumpProblem(joint_network, dproblem, Direct())
    solve(jump_problem, SSAStepper(), callback=cb, saveat=10)
end

function build_rate_functions(reaction_network::ReactionSystem)
    totalrate::Operation = sum((jumpratelaw(react, rxvars=[]) for react in reactions(reaction_network)))
    @show totalrate

    totalrate_fun = eval(build_function(totalrate, species(reaction_network), params(reaction_network); expression=Val{true}))

    symbol_map = speciesmap(reaction_network)
    rates = Dict{Vector{Int64},ModelingToolkit.Operation}()
    for react in reactions(reaction_network)
        net_change = zeros(Int64, numspecies(reaction_network))
        for (species, change) in react.netstoich
            net_change[symbol_map[species]] = change
        end

        rate = jumpratelaw(react, rxvars=[])
        
        if haskey(rates, net_change)
            rates[net_change] += rate
        else
            rates[net_change] = rate
        end
    end

    for key in keys(rates)
        rates[key] = log(rates[key])
    end

    @show rates
    rate_funs = [
        change => eval(build_function(expr, species(reaction_network), params(reaction_network))) for (change, expr) in pairs(rates)
    ]
    (totalrate_fun, rate_funs)
end

function build_rate_function_derivatives(reaction_network::ReactionSystem)
    totalrate::Operation = sum((jumpratelaw(react, rxvars=[]) for react in reactions(reaction_network)))

    println(totalrate)

    interaction_param = params(reaction_network)[1]()
    @derivatives D'~interaction_param

    d_totalrate = expand_derivatives(D(totalrate))

    println(d_totalrate)

    totalrate_fun = eval(build_function(d_totalrate, species(reaction_network), params(reaction_network); expression=Val{true}))

    symbol_map = speciesmap(reaction_network)
    rates = Dict{Vector{Int64},ModelingToolkit.Operation}()
    for react in reactions(reaction_network)
        net_change = zeros(Int64, numspecies(reaction_network))
        for (species, change) in react.netstoich
            net_change[symbol_map[species]] = change
        end

        rate = jumpratelaw(react, rxvars=[])
        
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
        change => eval(build_function(expr, species(reaction_network), params(reaction_network))) for (change, expr) in pairs(rates)
    ]
    (totalrate_fun, rate_funs)
end

function log_prob(solution::ODESolution, totalrate_fun, rate_funs; θ=1.0)
    result = 0.0
    for i in Base.Iterators.drop(eachindex(solution), 1)
        dt = solution.t[i] - solution.t[i - 1]
        u = solution[i - 1]
        result += - dt * totalrate_fun(u, θ)
    end
    
    for (net_change, rate_fun) in rate_funs
        du = solution[2] - solution[1]
        for i in Base.Iterators.drop(eachindex(solution), 1)
            du .= solution.u[i] .- solution.u[i - 1]
            u = solution.u[i - 1]

            if du == net_change
                result += rate_fun(u, θ)
            end
        end
    end

    result
end

d_totalrate, d_rates = build_rate_function_derivatives(joint_network)
r_totalrate, r_rates = build_rate_functions(response_network)

d_rates

r_rates

signal_response_pair(dproblem; θ=0.5).u

N_vals = 100_000
potentials = zeros(Float64, N_vals)
θ_vals = sort(rand(N_vals))
for (i, θ) in enumerate(θ_vals)
    sol = signal_response_pair(dproblem; θ=θ)
    ll = log_prob(sol, d_totalrate, d_rates; θ=θ) + log_prob(sol, r_totalrate, r_rates; θ=0.0)
    potentials[i] = ll
end

N′_vals = 1_000_000
lls = zeros(Float64, N′_vals)
for i in eachindex(lls)
    sol = signal_response_pair(dproblem; θ=1.0)
    ll = log_prob(sol, r_totalrate, r_rates; θ=1.0)
    lls[i] = ll
end

scatter(θ_vals, potentials)
histogram([lls, potentials])

mean(lls), std(lls) / sqrt(length(lls))
mean(potentials), std(potentials) / sqrt(length(potentials))

(mean(lls) - mean(potentials)) / 500
