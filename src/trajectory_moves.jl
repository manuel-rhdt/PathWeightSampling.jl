using DiffEqJump
using Catalyst
using Statistics
using Distributions
import Distributions: logpdf

include("trajectories/trajectory.jl")
include("trajectories/distribution.jl")
include("histogram_dist.jl")

struct StochasticSystem{uType,tType,R <: AbstractTrajectory{uType,tType},DP,J,P <: DiffEqBase.AbstractJumpProblem{DP,J},S <: UnivariateDistribution}
    jump_problem::P
    s0_dist::S
    distribution::TrajectoryDistribution
    response::R

    sparams::Vector{Float64}
    rparams::Vector{Float64}
end

mutable struct SignalChain{uType,tType,R <: AbstractTrajectory{uType,tType},DP,J,P <: DiffEqBase.AbstractJumpProblem{DP,J},S <: UnivariateDistribution}
    system::StochasticSystem{uType,tType,R,DP,J,P,S}
    # interaction parameter
    θ::Float64

    # to save statistics
    last_regrowth::Float64
    accepted_list::Vector{Float64}
    rejected_list::Vector{Float64}
end

chain(system::StochasticSystem, θ::Real) = SignalChain(system, θ, 0.0, Float64[], Float64[])

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

function new_signal(old_signal::Trajectory, system::StochasticSystem)
    jump_problem = system.jump_problem
    s0_dist = system.s0_dist
    sample = max(round(rand(s0_dist)), 0.0)
    u0 = SVector{1,Float64}(sample)

    tspan = (old_signal.t[begin], old_signal.t[end])
    jump_problem = myremake(jump_problem; u0=u0, tspan=tspan)
    new = solve(jump_problem, SSAStepper())
    Trajectory(SA[:S], new.t, new.u)
end

function propose!(new_signal::Trajectory, old_signal::Trajectory, chain::SignalChain)
    chain.last_regrowth = propose!(new_signal, old_signal, chain.system)
    new_signal
end

function propose!(new_signal::Trajectory, old_signal::Trajectory, system::StochasticSystem)
    jump_problem = system.jump_problem

    regrow_duration = rand() * duration(old_signal)

    if rand(Bool)
        shoot_forward!(new_signal, old_signal, jump_problem, old_signal.t[end] - regrow_duration)
    else
        shoot_backward!(new_signal, old_signal, jump_problem, old_signal.t[begin] + regrow_duration)
    end

    regrow_duration
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

energy(signal::Trajectory, chain::SignalChain) = energy(signal, chain.system, chain.θ)

function energy(signal::Trajectory, system::StochasticSystem, θ::Real)
    if θ == zero(θ)
        return 0.0
    end

    response = system.response
    joint = merge(signal, response)

    -θ * logpdf(system.distribution, joint, params=system.rparams)
end

struct ConfigurationGenerator
    sparams::Vector{Float64}
    rparams::Vector{Float64}
    distribution::TrajectoryDistribution
    signal_j_problem::JumpProblem
    joint_j_problem::JumpProblem
    s0_dist
    p0_dist
end

function configuration_generator(sn::ReactionSystem, rn::ReactionSystem, sparams, rparams, s_mean::Real, x_mean::Real)
    u0 = SVector{2,Float64}(s_mean, x_mean)

    joint_network = Base.merge(sn, rn)
    tspan = (0., 1.)
    discrete_prob = DiscreteProblem(joint_network, u0, tspan, vcat(sparams, rparams))
    discrete_prob = remake(discrete_prob, u0=u0)
    joint_p = JumpProblem(joint_network, discrete_prob, Direct())

    s0_dist, p0_dist = generate_stationary_distributions(joint_p, u0, 100 * 10_000)

    log_p0 = (s, x) -> if isinf(begin v = logpdf(p0_dist, [s, x]) end) v else v - logpdf(s0_dist, s) end

    u0s = SVector(s_mean)
    dprob_s = DiscreteProblem(sn, u0s, tspan, sparams)
    dprob_s = remake(dprob_s, u0=u0s)
    signal_p = JumpProblem(sn, dprob_s, Direct())

    ConfigurationGenerator(sparams, rparams, distribution(rn, log_p0), signal_p, joint_p, s0_dist, p0_dist)
end

function configuration_generator(sn::ReactionSystem, rn::ReactionSystem, sparams, rparams, s0_dist, p0_dist)
    log_p0 = (s, x) -> logpdf(p0_dist, [s, x]) - logpdf(s0_dist, s)

    joint_network = Base.merge(sn, rn)

    sample = max.(round.(rand(p0_dist)), 0.0)
    u0 = SVector(sample...)
    tspan = (0., 1.)
    discrete_prob = DiscreteProblem(joint_network, u0, tspan, vcat(sparams, rparams))
    discrete_prob = remake(discrete_prob, u0=SVector(discrete_prob.u0...))
    joint_p = JumpProblem(joint_network, discrete_prob, Direct())

    u0s = SVector(0.0)
    dprob_s = DiscreteProblem(sn, u0s, tspan, sparams)
    dprob_s = remake(dprob_s, u0=u0s)
    signal_p = JumpProblem(sn, dprob_s, Direct())

    ConfigurationGenerator(sparams, rparams, distribution(rn, log_p0), signal_p, joint_p, s0_dist, p0_dist)
end

function generate_configuration(gen::ConfigurationGenerator; duration::Real=500.0)
    p0_dist = gen.p0_dist
    sample = max.(round.(rand(p0_dist)), 0.0)
    u0 = SVector(sample...)

    jump_prob = myremake(gen.joint_j_problem, u0=u0, tspan=(0.0, duration))    
    sol = solve(jump_prob, SSAStepper())

    response = convert(Trajectory, trajectory(sol, SA[:X], SA[2]))
    signal = convert(Trajectory, trajectory(sol, SA[:S], SA[1]))

    (StochasticSystem(gen.signal_j_problem, gen.s0_dist, gen.distribution, response, gen.sparams, gen.rparams), signal)
end

function generate_configuration(sn::ReactionSystem, rn::ReactionSystem, sparams=[], rparams=[]; duration::Real=500.0)
    cg = configuration_generator(sn, rn, sparams, rparams)
    generate_configuration(cg, duration=duration)
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
