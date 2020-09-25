using Plots
using DifferentialEquations
using Catalyst
using ModelingToolkit
using Statistics

include("trajectories/trajectory.jl")
include("trajectories/distribution.jl")

mutable struct StochasticSystem{uType,tType,TRate,Rates,R <: AbstractTrajectory{uType,tType},DP,J,P <: DiffEqBase.AbstractJumpProblem{DP,J}}
    jump_problem::P
    distribution::TrajectoryDistribution{TRate,Rates}
    response::R
    # interaction parameter
    θ::Float64
end

function propose!(new_signal::Trajectory, old_signal::Trajectory, system::StochasticSystem)
    jump_problem = system.jump_problem
    if rand(Bool)
        shoot_forward!(new_signal, old_signal, jump_problem)
    else
        shoot_backward!(new_signal, old_signal, jump_problem)
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

function shoot_forward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::JumpProblem)
    num_steps = length(old_traj)
    branch_point = rand(2:num_steps - 1)

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

function shoot_backward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::JumpProblem)
    num_steps = length(old_traj)
    branch_point = rand(2:num_steps - 1)

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

struct ConfigurationGenerator{TRate,Rates}
    signal_network::ReactionSystem
    response_network::ReactionSystem
    joint_network::ReactionSystem
    distribution::TrajectoryDistribution{TRate,Rates}
end

function configuration_generator(sn::ReactionSystem, rn::ReactionSystem)
    ConfigurationGenerator(sn, rn, Base.merge(sn, rn), distribution(rn))
end

function generate_configuration(gen::ConfigurationGenerator; θ=1.0, duration::Float64=500.0)
    u0 = SVector(50, 50)
    tspan = (0., duration)
    discrete_prob = DiscreteProblem(u0, tspan)
    jump_prob = JumpProblem(gen.joint_network, discrete_prob, Direct())
    sol = solve(jump_prob, SSAStepper())

    response = convert(Trajectory, trajectory(sol, SA[:X], SA[2]))
    signal = convert(Trajectory, trajectory(sol, SA[:S], SA[1]))

    u0s = SVector(50)
    dprob_s = DiscreteProblem(u0s, tspan)
    jprob_s = JumpProblem(gen.signal_network, dprob_s, Direct())

    (StochasticSystem(jprob_s, gen.distribution, response, θ), signal)
end
