using Plots
using DifferentialEquations
using Catalyst
using ModelingToolkit
using Statistics

include("trajectories/trajectory.jl")
include("trajectories/distribution.jl")

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

struct StochasticConfiguration{uType,tType}
    jump_system::ReactionSystem
    distribution::TrajectoryDistribution
    response::Trajectory{uType,tType}
    signal::Trajectory{uType,tType}
    # interaction parameter
    θ::Float64
end

function Base.copy(conf::StochasticConfiguration)
    StochasticConfiguration(conf.jump_system, conf.distribution, copy(conf.response), copy(conf.signal), conf.θ)
end

function with_interaction(conf::StochasticConfiguration, θ)
    StochasticConfiguration(conf.jump_system, conf.distribution, copy(conf.response), copy(conf.signal), θ)
end

function propose!(new_conf::StochasticConfiguration, old_conf::StochasticConfiguration)
    jump_system = old_conf.jump_system
    if rand(Bool)
        shoot_forward!(new_conf.signal, old_conf.signal, jump_system)
    else
        shoot_backward!(new_conf.signal, old_conf.signal, jump_system)
    end
    nothing
end

function shoot_forward!(new_traj::Trajectory, old_traj::Trajectory, jump_system::ReactionSystem)
    num_steps = size(old_traj, 2)
    branch_point = rand(2:num_steps - 1)

    branch_time = old_traj.t[branch_point]
    branch_value = old_traj.u[:, branch_point]

    tspan = (branch_time, old_traj.t[end])

    dprob = DiscreteProblem(jump_system, branch_value, tspan)
    jprob = JumpProblem(jump_system, dprob, Direct())

    new_branch = solve(jprob, SSAStepper())

    new_traj.u = hcat(old_traj.u[:, begin:branch_point - 1], new_branch[:, :])
    new_traj.t = vcat(old_traj.t[begin:branch_point - 1], new_branch.t)
    nothing
end

function shoot_backward!(new_traj::Trajectory, old_traj::Trajectory, jump_system::ReactionSystem)
    num_steps = size(old_traj, 2)
    branch_point = rand(2:num_steps - 1)

    branch_time = old_traj.t[branch_point]
    branch_value = old_traj.u[:, branch_point]

    tspan = (old_traj.t[begin], branch_time)

    dprob = DiscreteProblem(jump_system, branch_value, tspan)
    jprob = JumpProblem(jump_system, dprob, Direct())

    new_branch = solve(jprob, SSAStepper())

    new_traj.u = hcat(new_branch[:, end:-1:begin], old_traj.u[:, branch_point + 1:end])
    new_traj.t = vcat(branch_time .- new_branch.t[end:-1:begin], old_traj.t[branch_point + 1:end])
    nothing
end

function energy(conf::StochasticConfiguration; θ=conf.θ)
    signal = conf.signal
    response = conf.response

    joint = LockstepIter(signal, response)

    -θ * logpdf(conf.distribution, joint)
end

function generate_initial_configuration(sn::ReactionSystem, rn::ReactionSystem, θ=1.0)
    joint_network = merge(sn, rn)

    u0 = [50, 50]
    tspan = (0., 500.)
    discrete_prob = DiscreteProblem(joint_network, u0, tspan)
    jump_prob = JumpProblem(joint_network, discrete_prob, Direct())
    sol = solve(jump_prob, SSAStepper())

    dist = distribution(rn)
    response = trajectory(sol, [:X])
    signal = trajectory(sol, [:S])

    StochasticConfiguration(sn, dist, response, signal, θ)
end

conf = generate_initial_configuration(sn, rn, 0.0)

with_interaction(conf, 1.0)