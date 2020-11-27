using StaticArrays

include("trajectories/trajectory.jl")
include("trajectories/distribution.jl")
include("histogram_dist.jl")

import DiffEqBase
import DiffEqJump: Direct, SSAStepper, JumpProblem

struct JumpSystem{TD <: TrajectoryDistribution,SP <: DiffEqBase.AbstractJumpProblem,RP <: DiffEqBase.AbstractJumpProblem,S <: UnivariateDistribution,P0 <: MultivariateDistribution}
    sparams::Vector{Float64}
    rparams::Vector{Float64}
    distribution::TD
    signal_j_problem::SP
    joint_j_problem::RP
    s0_dist::S
    p0_dist::P0
    duration::Float64
end

struct JumpSystemConfiguration{Signal <: Trajectory,Response <: Trajectory}
    signal::Signal
    response::Response
end

Base.copy(conf::JumpSystemConfiguration) = JumpSystemConfiguration(copy(conf.signal), copy(conf.response))

mutable struct SignalChain{Sys <: JumpSystem} <: MarkovChain
    system::Sys
    # interaction parameter
    θ::Float64

    # to save statistics
    last_regrowth::Float64
    accepted_list::Vector{Float64}
    rejected_list::Vector{Float64}
end

chain(system::JumpSystem; θ::Real=1.0) = SignalChain(system, θ, 0.0, Float64[], Float64[])

# reset statistics
function reset(pot::SignalChain)
    resize!(pot.accepted_list, 0)
    resize!(pot.rejected_list, 0)
end

function accept(pot::SignalChain)
    push!(pot.accepted_list, pot.last_regrowth)
end

function reject(pot::SignalChain)
    push!(pot.rejected_list, pot.last_regrowth)
end

new_signal(old::JumpSystemConfiguration, system::JumpSystem) = JumpSystemConfiguration(new_signal(old.signal, system), old.response)

function new_signal(old_signal::Trajectory, system::JumpSystem)
    jump_problem = system.signal_j_problem
    s0_dist = system.s0_dist
    sample = rand(s0_dist)
    u0 = SVector{1,Float64}(sample)

    tspan = (old_signal.t[begin], old_signal.t[end])
    jump_problem = myremake(jump_problem; u0=u0, tspan=tspan)
    new = solve(jump_problem, SSAStepper())
    Trajectory(SA[:S], new.t, new.u)
end

function propose!(new_conf::JumpSystemConfiguration, old_conf::JumpSystemConfiguration, chain::SignalChain)
    new_signal = propose!(new_conf.signal, old_conf.signal, chain)
    new_conf
end

function propose!(new_signal::Trajectory, old_signal::Trajectory, chain::SignalChain)
    chain.last_regrowth = propose!(new_signal, old_signal, chain.system)
    new_signal
end

function propose!(new_signal::Trajectory, old_signal::Trajectory, system::JumpSystem)
    jump_problem = system.signal_j_problem

    regrow_duration = rand() * duration(old_signal)

    if rand(Bool)
        shoot_forward!(new_signal, old_signal, jump_problem, old_signal.t[end] - regrow_duration)
    else
        shoot_backward!(new_signal, old_signal, jump_problem, old_signal.t[begin] + regrow_duration)
    end

    regrow_duration
end

function myremake(jprob::DiffEqBase.AbstractJumpProblem; u0, tspan)
    dprob = jprob.prob
    new_dprob = remake(dprob, u0=u0, tspan=tspan)
    JumpProblem(new_dprob, 
        Direct(),
        jprob.massaction_jump
    )
end

function shoot_forward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::DiffEqBase.AbstractJumpProblem, branch_time::Real)
    branch_value = old_traj(branch_time)
    branch_point = searchsortedfirst(old_traj.t, branch_time)
    tspan = (branch_time, old_traj.t[end])

    empty!(new_traj.u)
    empty!(new_traj.t)
    append!(new_traj.u, @view old_traj.u[begin:branch_point - 1])
    append!(new_traj.t, @view old_traj.t[begin:branch_point - 1])

    jump_problem = myremake(jump_problem; u0=branch_value, tspan=tspan)
    new_branch = solve(jump_problem, SSAStepper())
    
    append!(new_traj.t, new_branch.t)
    append!(new_traj.u, new_branch.u)
    nothing
end

function shoot_backward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::DiffEqBase.AbstractJumpProblem, branch_time::Real)
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

energy(configuration::JumpSystemConfiguration, chain::SignalChain) = energy(configuration, chain.system, chain.θ)

function energy(configuration::JumpSystemConfiguration, system::JumpSystem, θ::Real)
    signal = configuration.signal
    response = configuration.response
    joint = merge(signal, response)

    pot = logpdf(system.distribution, joint, params=system.rparams)
    if pot == -Inf
        # if we sample an impossible trajectory, return infinite energy, even at θ=0
        return -pot
    end

    -θ * pot
end

energy_difference(configuration::JumpSystemConfiguration, system::JumpSystem) = energy(configuration, system, 1.0)

function JumpSystem(sn::ReactionSystem, rn::ReactionSystem, sparams, rparams, s_mean::Real, x_mean::Real, duration::Real)
    u0 = SVector{2,Float64}(s_mean, x_mean)

    joint_network = Base.merge(sn, rn)
    tspan = (0., 1.)
    discrete_prob = DiscreteProblem(joint_network, u0, tspan, vcat(sparams, rparams))
    discrete_prob = remake(discrete_prob, u0=u0)
    joint_p = JumpProblem(joint_network, discrete_prob, Direct())

    s0_dist, p0_dist = generate_stationary_distributions(joint_p, u0, 100_000.0)

    log_p0 = (s, x) -> if isinf(begin v = logpdf(p0_dist, [s, x]) end) v else v - logpdf(s0_dist, s) end

    u0s = SVector(s_mean)
    dprob_s = DiscreteProblem(sn, u0s, tspan, sparams)
    dprob_s = remake(dprob_s, u0=u0s)
    signal_p = JumpProblem(sn, dprob_s, Direct())

    JumpSystem(sparams, rparams, distribution(rn, log_p0), signal_p, joint_p, s0_dist, p0_dist, duration)
end

function JumpSystem(sn::ReactionSystem, rn::ReactionSystem, sparams, rparams, s0_dist::UnivariateDistribution, p0_dist::MultivariateDistribution, duration::Real)
    log_p0 = (s, x) -> logpdf(p0_dist, [s, x]) - logpdf(s0_dist, s)

    joint_network = Base.merge(sn, rn)

    sample = rand(p0_dist)
    u0 = SVector(sample...)
    tspan = (0., 1.)
    discrete_prob = DiscreteProblem(joint_network, u0, tspan, vcat(sparams, rparams))
    discrete_prob = remake(discrete_prob, u0=SVector(discrete_prob.u0...))
    joint_p = JumpProblem(joint_network, discrete_prob, Direct())

    u0s = SVector(0.0)
    dprob_s = DiscreteProblem(sn, u0s, tspan, sparams)
    dprob_s = remake(dprob_s, u0=u0s)
    signal_p = JumpProblem(sn, dprob_s, Direct())

    JumpSystem(sparams, rparams, distribution(rn, log_p0), signal_p, joint_p, s0_dist, p0_dist, duration)
end

function generate_configuration(gen::JumpSystem)
    p0_dist = gen.p0_dist
    sample = rand(p0_dist)
    u0 = SVector(sample...)

    jump_prob = myremake(gen.joint_j_problem, u0=u0, tspan=(0.0, gen.duration))    
    sol = solve(jump_prob, SSAStepper())

    response = convert(Trajectory, trajectory(sol, SA[:X], SA[2]))
    signal = convert(Trajectory, trajectory(sol, SA[:S], SA[1]))

    JumpSystemConfiguration(signal, response)
end


function generate_stationary_distributions(jump_problem, u0, duration::Real)
    jump_problem = myremake(jump_problem, u0=u0, tspan=(0.0, duration))
    sol = solve(jump_problem, SSAStepper())

    weights = diff(sol.t)
    copy_numbers = sol.u[begin:end - 1]
    signal_copy_numbers = getindex.(copy_numbers, 1)

    joint_dist = mv_histogram_dist(copy_numbers, weights)
    signal_dist = histogram_dist(signal_copy_numbers, weights)

    signal_dist, joint_dist
end