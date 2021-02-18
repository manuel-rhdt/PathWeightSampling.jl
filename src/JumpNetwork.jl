import Distributions: logpdf

struct SRXsystem
    sn::ReactionSystem
    rn::ReactionSystem
    xn::ReactionSystem

    u0::AbstractVector

    ps::AbstractVector
    pr::AbstractVector
    px::AbstractVector

    dtimes
end

tspan(sys::SRXsystem) = (first(sys.dtimes), last(sys.dtimes))

function _solve(system::SRXsystem)
    joint = merge(merge(system.sn, system.rn), system.xn)
    u0 = SVector(system.u0...)

    tp = tspan(system)
    p = vcat(system.ps, system.pr, system.px)
    dprob = DiscreteProblem(joint, u0, tp, p)
    dprob = remake(dprob, u0=u0)
    jprob = JumpProblem(joint, dprob, Direct())

    sol = solve(jprob, SSAStepper())
end

function generate_configuration(system::SRXsystem)
    # we first generate a joint SRX trajectory
    joint = merge(merge(system.sn, system.rn), system.xn)
    sol = _solve(system)

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
    dtimes::Vector{Float64}
end

function MarginalEnsemble(system::SRXsystem)
    sr_network = merge(system.sn, system.rn)
    joint = merge(sr_network, system.xn)
    sr_idxs = species_indices(joint, Catalyst.species(sr_network)...)

    dprob = DiscreteProblem(sr_network, system.u0[sr_idxs], tspan(system), vcat(system.ps, system.pr))
    # we have to remake the discrete problem with u0 as StaticArray for 
    # improved performance
    dprob = remake(dprob, u0=SVector(dprob.u0...))
    jprob = JumpProblem(sr_network, dprob, Direct(), save_positions=(false, false))

    dep_species = dependent_species(system.xn)
    dep_idxs = species_indices(sr_network, dep_species...)

    MarginalEnsemble(jprob, distribution(system.xn), dep_idxs, system.px, collect(system.dtimes))
end
mutable struct TrajectoryCallback{uType, tType, N}
    traj::Trajectory{uType, tType, N}
    index::Int
end

TrajectoryCallback(traj::Trajectory) = TrajectoryCallback(traj, 1)

function (tc::TrajectoryCallback)(integrator::DiffEqBase.DEIntegrator) # affect!
    traj = tc.traj
    cond_u = traj.u[tc.index]
    tc.index = min(tc.index + 1, length(tc.traj.t))
    for i in eachindex(cond_u)
        integrator.u = setindex(integrator.u, cond_u[i], i)
    end
    # it is important to call this to properly update reaction rates
    DiffEqJump.reset_aggregated_jumps!(integrator, nothing, integrator.cb)
    nothing
end

function (tc::TrajectoryCallback)(u, t::Real, i::DiffEqBase.DEIntegrator)::Bool # condition
    while tc.index < length(tc.traj.t) && t > tc.traj.t[tc.index]
        tc.index += 1
    end
    t == tc.traj.t[tc.index]
end

struct ConditionalEnsemble{JP,XD,IX,DX}
    jump_problem::JP
    x_dist::XD
    indep_idxs::IX
    dep_idxs::DX
    xp::Vector{Float64}
    dtimes::Vector{Float64}
end

function ConditionalEnsemble(
    rn::ReactionSystem, 
    xn::ReactionSystem,
    u0,
    rp::Vector{Float64},
    xp::Vector{Float64},
    dtimes
)
    dprob = DiscreteProblem(rn, u0, (first(dtimes), last(dtimes)), rp)
    # we have to remake the discrete problem with u0 as StaticArray for 
    # improved performance
    dprob = remake(dprob, u0=SVector(dprob.u0...))
    jprob = JumpProblem(rn, dprob, Direct(), save_positions=(false, false))

    indep_species = independent_species(rn)
    indep_idxs = species_indices(rn, indep_species...)

    dep_species = dependent_species(xn)
    dep_idxs = SVector(indexin(species_indices(rn, dep_species...), indep_idxs))

    ConditionalEnsemble(jprob, distribution(xn), indep_idxs, dep_idxs, xp, collect(dtimes))
end

function ConditionalEnsemble(system::SRXsystem)
    joint = merge(merge(system.sn, system.rn), system.xn)
    r_idxs = species_indices(joint, Catalyst.species(system.rn)...)
    ConditionalEnsemble(system.rn, system.xn, system.u0[r_idxs], system.pr, system.px, system.dtimes)
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

function collect_samples(initial::SXconfiguration, system::MarginalEnsemble, num_samples::Int)
    jprob = system.jump_problem
    integrator = DiffEqBase.init(jprob, SSAStepper(), numsteps_hint=0)

    result = Array{Float64, 2}(undef, length(dtimes), num_samples)
    for result_col ∈ eachcol(result)
        integrator = DiffEqBase.init(jprob, SSAStepper(), numsteps_hint=0)
        iter = sub_trajectory(SSAIter(integrator), system.dep_idxs)
        cumulative_logpdf!(result_col, system.x_dist, merge_trajectories(iter, initial.x_traj), system.dtimes, params=system.xp)
    end

    result
end

function propagate(conf::SXconfiguration, ensemble::MarginalEnsemble, u0, tspan::Tuple)
    jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
    integrator = DiffEqBase.init(jprob, SSAStepper(), numsteps_hint=0)

    iter = sub_trajectory(SSAIter(integrator), ensemble.dep_idxs)

    log_weight = logpdf(ensemble.x_dist, merge_trajectories(iter, conf.x_traj), params=ensemble.xp)

    integrator.u, log_weight
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

function collect_samples(initial::SRXconfiguration, system::ConditionalEnsemble, num_samples::Int)
    cb = TrajectoryCallback(initial.s_traj)
    cb = DiscreteCallback(cb, cb, save_positions=(false, false))
    jprob = system.jump_problem

    idxs = system.indep_idxs[system.dep_idxs]

    result = Array{Float64, 2}(undef, length(dtimes), num_samples)
    for result_col ∈ eachcol(result)
        integrator = DiffEqBase.init(jprob, SSAStepper(), callback=cb, tstops=initial.s_traj.t, numsteps_hint=0)
        iter = SSAIter(integrator)
        dep = sub_trajectory(iter, idxs)
        cumulative_logpdf!(result_col, system.x_dist, merge_trajectories(dep, initial.x_traj), system.dtimes, params=system.xp)
    end

    result
end

function propagate(conf::SRXconfiguration, ensemble::ConditionalEnsemble, u0, tspan::Tuple)
    cb = TrajectoryCallback(conf.s_traj)
    cb = DiscreteCallback(cb, cb, save_positions=(false, false))
    jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
    integrator = DiffEqBase.init(jprob, SSAStepper(), callback=cb, tstops=conf.s_traj.t, numsteps_hint=0)

    idxs = ensemble.indep_idxs[ensemble.dep_idxs]
    iter = sub_trajectory(SSAIter(integrator), idxs)

    log_weight = logpdf(ensemble.x_dist, merge_trajectories(iter, conf.x_traj), params=ensemble.xp)

    integrator.u, log_weight
end

function energy_difference(configuration::SRXconfiguration, system::ConditionalEnsemble)
    dep = sub_trajectory(configuration.r_traj, system.dep_idxs)
    -logpdf(system.x_dist, merge_trajectories(dep, configuration.x_traj), params=system.xp)
end 

function marginal_configuration(conf::SRXconfiguration)
    new_s = collect_trajectory(merge_trajectories(conf.s_traj, conf.r_traj))
    SXconfiguration(new_s, conf.x_traj)
end
