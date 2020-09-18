using Plots
using DifferentialEquations
using Catalyst
using ModelingToolkit
using Statistics

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end


joint_network = merge(sn, rn)

signal_system = convert(JumpSystem, sn)

u0 = [50, 50]
tspan = (0., 500.)
discrete_prob = DiscreteProblem(joint_network, u0, tspan)
jump_prob = JumpProblem(joint_network, discrete_prob, Direct())
sol = solve(jump_prob, SSAStepper())
sol2 = solve(jump_prob, SSAStepper())

plot(sol2)

struct Trajectory{uType, tType}
    syms::Vector{Symbol}
    t::Vector{tType}
    u::Array{uType, 2}
end

function trajectory(sol::ODESolution)
    Trajectory(sol.prob.f.syms, sol.t, sol[:, :])
end

function trajectory(sol::ODESolution, syms::Vector{Symbol})
    idxs = UInt64[]
    for sym in syms
        i = findfirst(isequal(sym), sol.prob.f.syms)
        push!(idxs, i)
    end
    Trajectory(syms, sol.t, sol[idxs, :])
end

struct LockstepIter{uType, tType}
    first::Trajectory{uType, tType}
    second::Trajectory{uType, tType}
end

Base.iterate(iter::LockstepIter) = iterate(iter, (1, 1, min(iter.first.t[begin], iter.second.t[begin])))

function Base.iterate(iter::LockstepIter, (i, j, t))
    if i > size(iter.first.t, 1) && j > size(iter.second.t, 1)
        return nothing
    end

    current_t = t

    if (i+1) > size(iter.first.t, 1)
        t_i = Inf
    else 
        t_i = iter.first.t[i+1]
    end

    if (j+1) > size(iter.second.t, 1)
        t_j = Inf
    else
        t_j = iter.second.t[j+1]
    end

    u_1 = iter.first.u[:, i]
    u_2 = iter.second.u[:, j]

    if t_i < t_j
        t = t_i
        i = i + 1
    elseif t_j < t_i
        t = t_j
        j = j + 1
    else
        t = t_i
        i = i + 1
        j = j + 1
    end

    (vcat(u_1, u_2), current_t), (i, j, t)
end

traj = trajectory(sol)
traj2 = trajectory(sol2)

for x in LockstepIter(traj, traj2)
    @show x
end

tuples(sol)

trajectory(sol).u[:,1]

speciesmap(rn)

sol[:, :]

struct StochasticConfiguration{T,N,uType,uType2,DType,tType,rateType,P,A,IType,DE}
    jump_system::JumpSystem
    solution::ODESolution{T,N,uType,uType2,DType,tType,rateType,P,A,IType,DE}
end

function propose!(new_conf::StochasticConfiguration, old_conf::StochasticConfiguration)
    jump_system = old_conf.jump_system
    discrete_prob = old_conf.solution.prob
    shoot_forward!(new_conf.solution, old_conf.solution, jump_system, discrete_prob)
    nothing
end

function shoot_forward!(new_sol::ODESolution, old_sol::ODESolution, jump_system::JumpSystem, discrete_prob::DiscreteProblem)
    num_steps = size(old_sol, 2)
    branch_point = rand(2:num_steps)

    branch_time = old_sol.t[branch_point]
    branch_value = old_sol.u[branch_point]

    dprob =  remake(discrete_prob; tspan=(branch_time, discrete_prob.tspan[2]), u0=branch_value)
    new_jump_prob = JumpProblem(jump_system, dprob, Direct())

    new_branch = solve(new_jump_prob, SSAStepper())

    resize!(new_sol.u, 0)
    resize!(new_sol.t, 0)

    append!(new_sol.u, old_sol.u[begin:branch_point-1])
    append!(new_sol.t, old_sol.t[begin:branch_point-1])

    append!(new_sol.u, new_branch.u)
    append!(new_sol.t, new_branch.t)
    nothing
end

test_conf = StochasticConfiguration(signal_system, sol)
test_conf2 = deepcopy(test_conf)

p = plot(test_conf.solution)

propose!(test_conf2, test_conf)
plot!(p, test_conf2.solution)

# needed to work around issue in ModelingToolkit where the call to jumpratelaw fails
# if the rate is a simple number instead of an operation
ModelingToolkit.get_variables(f::Number) = Variable[]

function build_rate_functions(reaction_network::ReactionSystem)
    totalrate = sum((jumpratelaw(react) for react in reactions(reaction_network)))
    @show totalrate

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

    for key in keys(rates)
        rates[key] = log(rates[key])
    end

    @show rates
    rate_funs = [
        change => build_function(expr, species(reaction_network), params(reaction_network); expression=Val{false}) for (change, expr) in pairs(rates)
    ]
    (totalrate_fun, rate_funs)
end

(stotalrate, srates) = build_rate_functions(rn)
stotalrate([50, 30], [])
srates[2][2]([50, 30], [])

function log_probability(solution, totalrate_fun, rate_funs; params=[])
    result = 0.0
    for i in Base.Iterators.drop(eachindex(solution), 1)
        dt = solution.t[i] - solution.t[i - 1]
        u = solution[i - 1]
        result += - dt * totalrate_fun(u, params)
    end
    
    for (net_change, rate_fun) in rate_funs
        du = solution[2] - solution[1]
        for i in Base.Iterators.drop(eachindex(solution), 1)
            du .= solution.u[i] .- solution.u[i - 1]
            u = solution.u[i - 1]

            if du == net_change
                result += rate_fun(u, params)
            end
        end
    end

    result
end