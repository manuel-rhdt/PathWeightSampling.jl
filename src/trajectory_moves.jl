using DiffEqJump
using Catalyst
using Statistics
using Distributions
import Distributions: logpdf

include("trajectories/trajectory.jl")
include("trajectories/distribution.jl")

mutable struct StochasticSystem{uType,tType,R <: AbstractTrajectory{uType,tType},DP,J,P <: DiffEqBase.AbstractJumpProblem{DP,J}}
    jump_problem::P
    distribution::TrajectoryDistribution
    response::R
    # interaction parameter
    θ::Float64

    # to save statistics
    last_regrowth::Float64
    accepted_list::Vector{Float64}
    rejected_list::Vector{Float64}
end

# reset statistics
function reset(system::StochasticSystem)
    resize!(system.accepted_list, 0)
    resize!(system.rejected_list, 0)
end

function accept(system::StochasticSystem)
    push!(system.accepted_list, system.last_regrowth)
end

function reject(system::StochasticSystem)
    push!(system.rejected_list, system.last_regrowth)
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

    regrow_duration = rand() * duration(old_signal)

    if rand(Bool)
        shoot_forward!(new_signal, old_signal, jump_problem, old_signal.t[end] - regrow_duration)
    else
        shoot_backward!(new_signal, old_signal, jump_problem, old_signal.t[begin] + regrow_duration)
    end

    system.last_regrowth = regrow_duration
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

function shoot_forward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::JumpProblem, branch_time::Real)
    branch_value = old_traj(branch_time)
    branch_point = searchsortedfirst(old_traj.t, branch_time)
    tspan = (branch_time, old_traj.t[end])

    resize!(new_traj.u, 0)
    resize!(new_traj.t, 0)
    append!(new_traj.u, old_traj[begin:branch_point - 1])
    append!(new_traj.t, old_traj.t[begin:branch_point - 1])


    jump_problem = myremake(jump_problem; u0=branch_value, tspan=tspan)
    new_branch = solve(jump_problem, SSAStepper())

    # integrator = init(jump_problem, SSAStepper())
    # for (u, t) in tuples(integrator)
    #    
    # end

    
    append!(new_traj.t, new_branch.t)
    append!(new_traj.u, new_branch.u)
    nothing
end

function shoot_backward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::JumpProblem, branch_time::Real)
    branch_value = old_traj(branch_time)
    branch_point = searchsortedfirst(old_traj.t, branch_time)
    tspan = (old_traj.t[begin], branch_time)

    jump_problem = myremake(jump_problem; u0=branch_value, tspan=tspan)
    new_branch = solve(jump_problem, SSAStepper())

    resize!(new_traj.u, 0)
    resize!(new_traj.t, 0)

    append!(new_traj.u, @view new_branch.u[end - 1:-1:begin])
    append!(new_traj.u, @view old_traj.u[branch_point:end])

    append!(new_traj.t, branch_time .- @view new_branch.t[end:-1:begin + 1])
    append!(new_traj.t, @view old_traj.t[branch_point:end])
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
    p0_dist = MvNormal([50.0, 50.0], [50.0 100.0/3; 100.0/3 250.0/3])
    s0_dist = MvNormal([50.0], [50.0])

    log_p0 = (s, x) -> logpdf(p0_dist, [s, x]) - logpdf(s0_dist, [s])

    ConfigurationGenerator(sn, rn, Base.merge(sn, rn), distribution(rn, log_p0))
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

    (StochasticSystem(jprob_s, gen.distribution, response, θ, 0.0, Float64[], Float64[]), signal)
end

function generate_configuration(sn::ReactionSystem, rn::ReactionSystem; θ::Real=1.0, duration::Real=500.0)
    cg = configuration_generator(sn, rn)
    generate_configuration(cg, θ=θ, duration=duration)
end
