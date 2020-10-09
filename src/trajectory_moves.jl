using Plots
using DiffEqJump
using Catalyst
using ModelingToolkit
using Statistics

include("trajectories/trajectory.jl")
include("trajectories/distribution.jl")

mutable struct StochasticSystem{uType,tType,R <: AbstractTrajectory{uType,tType},DP,J,P <: DiffEqBase.AbstractJumpProblem{DP,J}}
    jump_problem::P
    distribution::TrajectoryDistribution
    response::R
    # interaction parameter
    θ::Float64
end

function new_signal(old_signal::Trajectory, system::StochasticSystem)
    jump_problem = system.jump_problem

    tspan = (old_signal.t[begin], old_signal.t[end])

    jump_problem = myremake(jump_problem; u0=old_signal.u[begin], tspan=tspan)
    new = solve(jump_problem, SSAStepper())
    Trajectory(SA[:S], new.t, new.u)
end

function propose!(new_signal::Trajectory, old_signal::Trajectory, system::StochasticSystem)
    jump_problem = system.jump_problem

    # we want to regenerate at least 1/4 of the trajectory
    num_steps = length(old_signal)
    at_least_one_quarter = div(num_steps, 4, RoundUp)
    branch_point = rand(at_least_one_quarter:num_steps - at_least_one_quarter)

    if rand(Bool)
        shoot_forward!(new_signal, old_signal, jump_problem, branch_point)
    else
        shoot_backward!(new_signal, old_signal, jump_problem, branch_point)
    end
    nothing
end

function myremake(jprob::JumpProblem; u0, tspan)
    dprob = jprob.prob
    new_dprob = remake(dprob, u0=u0, tspan=tspan)
    JumpProblem(new_dprob, 
        Direct(),
        jprob.massaction_jump
    )
end

function shoot_forward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::JumpProblem, branch_point::Int)
    branch_time = old_traj.t[branch_point]
    branch_value = old_traj[branch_point]

    tspan = (branch_time, old_traj.t[end])

    jump_problem = myremake(jump_problem; u0=branch_value, tspan=tspan)
    new_branch = solve(jump_problem, SSAStepper())

    resize!(new_traj.u, 0)
    resize!(new_traj.t, 0)

    append!(new_traj.u, old_traj[begin:branch_point - 1])
    append!(new_traj.u, new_branch.u)

    append!(new_traj.t, old_traj.t[begin:branch_point - 1])
    append!(new_traj.t, new_branch.t)
    nothing
end

function shoot_backward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::JumpProblem, branch_point::Int)
    branch_time = old_traj.t[branch_point]
    branch_value = old_traj[branch_point]

    tspan = (old_traj.t[begin], branch_time)

    jump_problem = myremake(jump_problem; u0=branch_value, tspan=tspan)
    new_branch = solve(jump_problem, SSAStepper())

    resize!(new_traj.u, 0)
    resize!(new_traj.t, 0)

    append!(new_traj.u, new_branch.u[end:-1:begin])
    append!(new_traj.u, old_traj[branch_point + 1:end])

    append!(new_traj.t, branch_time .- new_branch.t[end:-1:begin])
    append!(new_traj.t, old_traj.t[branch_point + 1:end])
    nothing
end

function energy(signal::Trajectory, system::StochasticSystem; θ=system.θ)
    response = system.response

    joint = merge(signal, response)

    -θ * logpdf(system.distribution, joint)
end

struct ConfigurationGenerator
    signal_network::ReactionSystem
    response_network::ReactionSystem
    joint_network::ReactionSystem
    distribution::TrajectoryDistribution
end

function configuration_generator(sn::ReactionSystem, rn::ReactionSystem)
    ConfigurationGenerator(sn, rn, Base.merge(sn, rn), distribution(rn))
end

function generate_configuration(gen::ConfigurationGenerator; θ=1.0, duration::Float64=500.0)
    u0 = SVector(50.0, 50.0)
    tspan = (0., duration)
    discrete_prob = DiscreteProblem(u0, tspan)
    jump_prob = JumpProblem(gen.joint_network, discrete_prob, Direct())
    sol = solve(jump_prob, SSAStepper())

    response = convert(Trajectory, trajectory(sol, SA[:X], SA[2]))
    signal = convert(Trajectory, trajectory(sol, SA[:S], SA[1]))

    u0s = SVector(50.0)
    dprob_s = DiscreteProblem(u0s, tspan)
    jprob_s = JumpProblem(gen.signal_network, dprob_s, Direct())

    (StochasticSystem(jprob_s, gen.distribution, response, θ), signal)
end
