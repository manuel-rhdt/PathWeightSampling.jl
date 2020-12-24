import Distributions: logpdf

struct SRXsystem
    sn::ReactionSystem
    rn::ReactionSystem
    xn::ReactionSystem

    u0::AbstractVector

    ps::AbstractVector
    pr::AbstractVector
    px::AbstractVector

    tspan
end

function generate_configuration(system::SRXsystem)
    # we first generate a joint SRX trajectory
    joint = merge(merge(system.sn, system.rn), system.xn)

    u0 = SVector(system.u0...)

    tspan = system.tspan
    p = vcat(system.ps, system.pr, system.px)
    dprob = DiscreteProblem(joint, u0, tspan, p)
    dprob = remake(dprob, u0=u0)
    jprob = JumpProblem(joint, dprob, Direct())

    sol = solve(jprob, SSAStepper())

    # then we extract the signal
    s_spec = independent_species(system.sn)
    s_idxs = species_indices(joint, s_spec...)
    s_traj = Trajectory(trajectory(sol, s_idxs))

    # the R trajectory
    r_spec = independent_species(system.rn)
    r_idxs = species_indices(joint, r_spec...)
    r_traj = Trajectory(trajectory(sol, r_idxs))
    
    # finally we extract the X part from the SRX trajectory
    x_spec = independent_species(system.xn)
    x_idxs = species_indices(joint, x_spec...)
    x_traj = Trajectory(trajectory(sol, x_idxs))

    SRXconfiguration(s_traj, r_traj, x_traj)
end

struct MarginalEnsemble{JP,XD,DX}
    jump_problem::JP
    x_dist::XD
    dep_idxs::DX
    xp::Vector{Float64}
end

function MarginalEnsemble(system::SRXsystem)
    sr_network = merge(system.sn, system.rn)
    joint = merge(sr_network, system.xn)
    sr_idxs = species_indices(joint, Catalyst.species(sr_network)...)

    dprob = DiscreteProblem(sr_network, system.u0[sr_idxs], system.tspan, vcat(system.ps, system.pr))
    # we have to remake the discrete problem with u0 as StaticArray for 
    # improved performance
    dprob = remake(dprob, u0=SVector(dprob.u0...))
    jprob = JumpProblem(sr_network, dprob, Direct(), save_positions=(false, false))

    dep_species = dependent_species(system.xn)
    dep_idxs = species_indices(sr_network, dep_species...)

    MarginalEnsemble(jprob, distribution(system.xn), dep_idxs, system.px)
end
struct TrajectoryCallback
    traj::Trajectory
end

function (tc::TrajectoryCallback)(integrator::DiffEqBase.DEIntegrator) # affect!
    traj = tc.traj
    cond_u = traj(integrator.t)
    for i in eachindex(cond_u)
        integrator.u = setindex(integrator.u, cond_u[i], i)
    end
    # it is important to call this to properly update reaction rates
    DiffEqJump.reset_aggregated_jumps!(integrator, nothing, integrator.cb)
    nothing
end

function (tc::TrajectoryCallback)(u, t::Real, i::DiffEqBase.DEIntegrator)::Bool # condition
    t ∈ tc.traj.t
end

struct ConditionalEnsemble{JP,XD,IX,DX}
    jump_problem::JP
    x_dist::XD
    indep_idxs::IX
    dep_idxs::DX
    xp::Vector{Float64}
end

function ConditionalEnsemble(
    rn::ReactionSystem, 
    xn::ReactionSystem,
    u0,
    rp::Vector{Float64},
    xp::Vector{Float64},
    tspan
)
    dprob = DiscreteProblem(rn, u0, tspan, rp)
    # we have to remake the discrete problem with u0 as StaticArray for 
    # improved performance
    dprob = remake(dprob, u0=SVector(dprob.u0...))
    jprob = JumpProblem(rn, dprob, Direct(), save_positions=(false, false))

    indep_species = independent_species(rn)
    indep_idxs = species_indices(rn, indep_species...)

    dep_species = dependent_species(xn)
    dep_idxs = SVector(indexin(species_indices(rn, dep_species...), indep_idxs))

    ConditionalEnsemble(jprob, distribution(xn), indep_idxs, dep_idxs, xp)
end

function ConditionalEnsemble(system::SRXsystem)
    joint = merge(merge(system.sn, system.rn), system.xn)
    r_idxs = species_indices(joint, Catalyst.species(system.rn)...)
    ConditionalEnsemble(system.rn, system.xn, system.u0[r_idxs], system.pr, system.px, system.tspan)
end

# returns a list of species in `a` that also occur in `b`
function intersecting_species(a::ReactionSystem, b::ReactionSystem)
    intersect(Catalyst.species(a), Catalyst.species(b))
end

# returns a list of species in `a` that are not in `b`
function unique_species(a::ReactionSystem, b::ReactionSystem)
    setdiff(Catalyst.species(a), Catalyst.species(b))
end

function species_indices(rs::ReactionSystem, species...)
    SVector(getindex.(Ref(Catalyst.speciesmap(rs)), species))
end

function independent_species(rs::ReactionSystem)
    i_spec = []
    for r in Catalyst.reactions(rs)
        push!(i_spec, getindex.(r.netstoich, 1)...)
    end
    unique(s for s∈i_spec)
end

function dependent_species(rs::ReactionSystem)
    setdiff(Catalyst.species(rs), independent_species(rs))
end

struct SXconfiguration{uType,tType,Ns,Nx}
    s_traj::Trajectory{uType,tType,Ns}
    x_traj::Trajectory{uType,tType,Nx}
end

function sample(configuration::T, system::MarginalEnsemble; θ=0.0)::T where T<:SXconfiguration
    if θ != 0.0
        error("can only use DirectMC with JumpNetwork")
    end
    jprob = system.jump_problem
    integrator = DiffEqBase.init(jprob, SSAStepper(), numsteps_hint=0)
    iter = SSAIter(integrator)
    s_traj = collect_trajectory(iter)
    SXconfiguration(s_traj, configuration.x_traj)
end

function energy_difference(configuration::SXconfiguration, system::MarginalEnsemble)
    dep = sub_trajectory(configuration.s_traj, system.dep_idxs)
    -logpdf(system.x_dist, merge_trajectories(dep, configuration.x_traj), params=system.xp)
end 
struct SRXconfiguration{uType,tType,Ns,Nr,Nx}
    s_traj::Trajectory{uType,tType,Ns}
    r_traj::Trajectory{uType,tType,Nr}
    x_traj::Trajectory{uType,tType,Nx}
end

function sample(configuration::T, system::ConditionalEnsemble; θ=0.0)::T where T<:SRXconfiguration
    if θ != 0.0
        error("can only use DirectMC with JumpNetwork")
    end
    cb = TrajectoryCallback(configuration.s_traj)
    cb = DiscreteCallback(cb, cb, save_positions=(false, false))
    jprob = system.jump_problem
    integrator = DiffEqBase.init(jprob, SSAStepper(), callback=cb, tstops=configuration.s_traj.t, numsteps_hint=0)
    iter = SSAIter(integrator)

    rtraj = collect_trajectory(sub_trajectory(iter, system.indep_idxs))
    SRXconfiguration(configuration.s_traj, rtraj, configuration.x_traj)
end

function energy_difference(configuration::SRXconfiguration, system::ConditionalEnsemble)
    dep = sub_trajectory(configuration.r_traj, system.dep_idxs)
    -logpdf(system.x_dist, merge_trajectories(dep, configuration.x_traj), params=system.xp)
end 

function marginal_configuration(conf::SRXconfiguration)
    new_s = collect_trajectory(merge_trajectories(conf.s_traj, conf.r_traj))
    SXconfiguration(new_s, conf.x_traj)
end

# To compute the MI we need the ability
# - create a new configuration (i.e. jointly sample S, R, X)
# - for a given configuration replace the R part of the trajectory
# - for a given configuration replace the S part of the trajectory
# - compute P(r, x | s)
# - compute P_0(r)

# system

# conf = generate_configuration(system)
# conf.r_ensemble.jump_problem.prob
# sig = sample(conf)

# samples = [sample(conf) for i ∈ 1:10000]
# e = energy_difference.(samples)

# using Plots
# histogram(e)

# using DiffEqBase, DiffEqJump

# u0 = SA[10, 30, 0, 50, 0, 0]
# tspan = (0.0, 10.0)
# ps = [5.0, 1.0]
# pr = [1.0, 4.0, 1.0, 2.0]
# px = [1.0, 1.0]

# system = SRXsystem(sn, rn, xn, u0, ps, pr, px, tspan)

# initial = generate_configuration(system)
# cond_ensemble = ConditionalEnsemble(system)
# marg_ensemble = MarginalEnsemble(system)

# marginal_configuration(initial)

# using Plots
# c = sample(initial, cond_ensemble)
# energy_difference(c, cond_ensemble)

# c = sample(marginal_configuration(initial), marg_ensemble)
# energy_difference(c, marg_ensemble)

# algorithm = DirectMCEstimate(10000)
# result = simulate(algorithm, initial, cond_ensemble)
# log_marginal(result)

# result2 = simulate(algorithm, marginal_configuration(initial), marg_ensemble)
# log_marginal(result2)