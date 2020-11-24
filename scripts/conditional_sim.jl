using Plots
using DifferentialEquations
using Catalyst
using ModelingToolkit
using GaussianMcmc.Trajectories

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

u0 = [50]
tspan = (0., 10.)
discrete_prob = DiscreteProblem(sn, u0, tspan)
jump_prob = JumpProblem(sn, discrete_prob, Direct())
sol = solve(jump_prob, SSAStepper())
signal = Trajectories.trajectory(sol)

plot(signal)
signal.u[3:end] .= [SVector{1}(0)]
signal.u

rn = @reaction_network begin
    0.01 * θ, S --> X + S
    0.01 * 50.0 * (1-θ), ∅ --> X
    0.01, X --> ∅ 
end θ

mutable struct SignalStepper
    signal::Trajectory
    index::Int
end

function (signal_stepper::SignalStepper)(integrator)
    if integrator.t == zero(integrator.t)
        signal_stepper.index = 1
    end
    integrator.u[1] = signal_stepper.signal.u[signal_stepper.index][1]
    signal_stepper.index += 1
end
stepper = SignalStepper(signal, 1)

signal_cb = PresetTimeCallback(sol.t, stepper)

u0r = [50, 50]
p = [1.0]
response_prob = DiscreteProblem(rn, u0r, tspan, p)
jump_prob_r = JumpProblem(rn, response_prob, Direct())
sol_r = solve(jump_prob_r, SSAStepper(), callback=signal_cb, tstops=sol.t)
plot(sol_r)

Trajectories.trajectory(sol_r)

react = reactions(rn)[1]
ratelaw = jumpratelaw(react)
s_rate = build_function(ratelaw, species(rn), params(rn))
s_ratefun = eval(s_rate)
s_ratefun(0.5, 1.0)

sol_r.t[end]
sol_r[end]


function build_rate_functions(rn::ReactionSystem)
    totalrate = sum((jumpratelaw(react, rxvars=[]) for react in reactions(rn)))
    print(totalrate)
    totalrate_fun = eval(build_function(totalrate, species(rn), params(rn), expression=Val{true}))
    rate_funs = [eval(build_function(jumpratelaw(react, rxvars=[]), species(rn), params(rn), expression=Val{true})) for react in reactions(rn)]
    (totalrate_fun, rate_funs)
end

function log_likelihood(solution::ODESolution, rn::ReactionSystem, totalrate_fun, rate_funs; θ=1.0)
    result = 0.0
    for i in Base.Iterators.drop(eachindex(solution), 1)
        dt = solution.t[i] - solution.t[i - 1]
        u = solution[i - 1]
        result += - dt * totalrate_fun(u, θ)
    end

    symbol_map = speciesmap(rn)
    
    reaction_props = zeros(length(solution) - 1)
    for (rate_fun, react) in zip(rate_funs, reactions(rn))
        net_change = zeros(Int64, numspecies(rn))
        for (species, change) in react.netstoich
            net_change[symbol_map[species]] = change
        end

        
        du = solution[2] - solution[1]
        for i in Base.Iterators.drop(eachindex(solution), 1)
            du .= solution.u[i] .- solution.u[i - 1]
            u = solution.u[i - 1]

            if du == net_change
                reaction_props[i-1] += rate_fun(u, θ) / totalrate_fun(u, θ)
            end
        end
    end

    for r in reaction_props
        if r > 0.0
            result += log(r)
        end
    end

    result
end

totalrate, rates = build_rate_functions(rn)
log_likelihood(sol_r, rn, totalrate, rates; θ=0.0)

totalrate([54, 50], 0.99)

function signal_response_pair(signal_problem, response_problem; θ=1.0)
    signal = solve(signal_problem, SSAStepper())
    signal_cb = PresetTimeCallback(signal.t, (integrator) -> integrator.u[1] = signal(integrator.t)[1])

    problem = remake(response_problem; p=[θ])
    response = solve(response_problem, SSAStepper(), callback=signal_cb, tstops=signal.t)
    response
end

result = signal_response_pair(jump_prob, jump_prob_r, θ=0.0)
log_likelihood(result, rn, totalrate, rates, 0.0)
plot(result)

ll_vals = Vector{Float64}()
θ_vals = rand(100000)
for θ in θ_vals
    result = signal_response_pair(jump_prob, jump_prob_r, θ=θ)
    ll = log_likelihood(result, rn, totalrate, rates; θ=θ)
    push!(ll_vals, ll)
end

scatter(θ_vals, ll_vals)

ll_vals_cond = Vector{Float64}()
for θ in ones(100000)
    result = signal_response_pair(jump_prob, jump_prob_r, θ=θ)
    ll = log_likelihood(result, rn, totalrate, rates, θ)
    push!(ll_vals_cond, ll)
end

histogram([ll_vals_cond, ll_vals], alpha=0.5)

mean(ll_vals) - mean(ll_vals_cond)S