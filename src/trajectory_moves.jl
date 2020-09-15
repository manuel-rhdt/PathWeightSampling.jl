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
end θ


joint_network = merge(sn, rn)

signal_system = convert(JumpSystem, sn)

u0 = [50]
tspan = (0., 500.)
discrete_prob = DiscreteProblem(sn, u0, tspan)
jump_prob = JumpProblem(signal_system, discrete_prob, Direct())
sol = solve(jump_prob, SSAStepper())

plot(sol)

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
