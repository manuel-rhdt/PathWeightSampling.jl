using DiffEqJump
using Catalyst
using Statistics
using Distributions
import Distributions: logpdf

include("trajectories/trajectory.jl")
include("trajectories/distribution.jl")

mutable struct StochasticSystem{uType,tType,R <: AbstractTrajectory{uType,tType},DP,J,P <: DiffEqBase.AbstractJumpProblem{DP,J}}
    jump_problem::P
    s0_dist::MultivariateNormal
    distribution::TrajectoryDistribution
    response::R
    # interaction parameter
    θ::Float64

    # to save statistics
    last_regrowth::Float64
    accepted_list::Vector{Float64}
    rejected_list::Vector{Float64}

    params::Vector{Float64}
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
    s0_dist = system.s0_dist
    sample = max.(round.(rand(s0_dist)), 0.0)
    u0 = SVector(sample...)

    tspan = (old_signal.t[begin], old_signal.t[end])
    jump_problem = myremake(jump_problem; u0=u0, tspan=tspan)
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

    empty!(new_traj.u)
    empty!(new_traj.t)
    append!(new_traj.u, @view old_traj.u[begin:branch_point - 1])
    append!(new_traj.t, @view old_traj.t[begin:branch_point - 1])


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

    empty!(new_traj.u)
    empty!(new_traj.t)

    append!(new_traj.u, @view new_branch.u[end - 1:-1:begin])
    append!(new_traj.u, @view old_traj.u[branch_point:end])

    for rtime in @view new_branch.t[end:-1:begin + 1]
        push!(new_traj.t, branch_time - rtime)
    end
    append!(new_traj.t, @view old_traj.t[branch_point:end])
    nothing
end

function energy(signal::Trajectory, system::StochasticSystem; θ=system.θ)
    response = system.response
    joint = merge(signal, response)

    -θ * logpdf(system.distribution, joint, params=system.params)
end

struct ConfigurationGenerator
    signal_network::ReactionSystem
    response_network::ReactionSystem
    sparams::Vector{Float64}
    rparams::Vector{Float64}
    joint_network::ReactionSystem
    distribution::TrajectoryDistribution
    s0_dist::MultivariateNormal
    p0_dist::MultivariateNormal
end

function configuration_generator(sn::ReactionSystem, rn::ReactionSystem, sparams, rparams, s0_dist, p0_dist)
    log_p0 = (s, x) -> logpdf(p0_dist, [s, x]) - logpdf(s0_dist, [s])

    ConfigurationGenerator(sn, rn, sparams, rparams, Base.merge(sn, rn), distribution(rn, log_p0), s0_dist, p0_dist)
end

function generate_configuration(gen::ConfigurationGenerator; θ=1.0, duration::Float64=500.0)
    p0_dist = gen.p0_dist
    sample = max.(round.(rand(p0_dist)), 0.0)

    u0 = SVector(sample...)
    tspan = (0., duration)
    discrete_prob = DiscreteProblem(gen.joint_network, u0, tspan, vcat(gen.sparams, gen.rparams))
    discrete_prob = remake(discrete_prob, u0=SVector(discrete_prob.u0...))
    jump_prob = JumpProblem(gen.joint_network, discrete_prob, Direct())
    sol = solve(jump_prob, SSAStepper())

    response = convert(Trajectory, trajectory(sol, SA[:X], SA[2]))
    signal = convert(Trajectory, trajectory(sol, SA[:S], SA[1]))

    u0s = SVector(0.0)
    dprob_s = DiscreteProblem(gen.signal_network, u0s, tspan, gen.sparams)
    dprob_s = remake(dprob_s, u0=u0s)
    jprob_s = JumpProblem(gen.signal_network, dprob_s, Direct())

    (StochasticSystem(jprob_s, gen.s0_dist, gen.distribution, response, θ, 0.0, Float64[], Float64[], gen.rparams), signal)
end

function generate_configuration(sn::ReactionSystem, rn::ReactionSystem, sparams=[], rparams=[]; θ::Real=1.0, duration::Real=500.0)
    cg = configuration_generator(sn, rn, sparams, rparams)
    generate_configuration(cg, θ=θ, duration=duration)
end
