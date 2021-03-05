import Distributions:logpdf

struct MarginalEnsemble{JP<:DiffEqBase.AbstractJumpProblem,XD}
    jump_problem::JP
    x_dist::XD
    xp::Vector{Float64}
    dtimes::Vector{Float64}
end

struct ConditionalEnsemble{JP<:DiffEqBase.AbstractJumpProblem,XD,IX,DX}
    jump_problem::JP
    x_dist::XD
    indep_idxs::IX
    dep_idxs::DX
    xp::Vector{Float64}
    dtimes::Vector{Float64}
end

struct SXconfiguration{uType,tType}
    s_traj::Trajectory{uType,tType}
    x_traj::Trajectory{uType,tType}
end
struct SRXconfiguration{uType,tType}
    s_traj::Trajectory{uType,tType}
    r_traj::Trajectory{uType,tType}
    x_traj::Trajectory{uType,tType}
end

abstract type JumpNetwork end
struct SXsystem <: JumpNetwork
    sn::ReactionSystem
    xn::ReactionSystem

    u0::AbstractVector

    ps::AbstractVector
    px::AbstractVector

    dtimes

    jump_problem
end

function SXsystem(sn, xn, u0, ps, px, dtimes)
    joint = merge(sn, xn)

    tp = (first(dtimes), last(dtimes))
    p = vcat(ps, px)
    dprob = DiscreteProblem(joint, copy(u0), tp, p)
    jprob = JumpProblem(joint, dprob, Direct())

    SXsystem(sn, xn, u0, ps, px, dtimes, jprob)
end

struct CompiledSXsystem{JP, XD}
    system::SXsystem
    marginal_ensemble::MarginalEnsemble{JP, XD}
end

compile(s::SXsystem) = CompiledSXsystem(s, MarginalEnsemble(s))
marginal_density(csx::CompiledSXsystem, algorithm, conf::SXconfiguration) = log_marginal(simulate(algorithm, conf, csx.marginal_ensemble))
conditional_density(csx::CompiledSXsystem, algorithm, conf::SXconfiguration) = -energy_difference(conf, csx.marginal_ensemble)

struct SRXsystem <: JumpNetwork
    sn::ReactionSystem
    rn::ReactionSystem
    xn::ReactionSystem

    u0::AbstractVector

    ps::AbstractVector
    pr::AbstractVector
    px::AbstractVector

    dtimes

    jump_problem
end

function SRXsystem(sn, rn, xn, u0, ps, pr, px, dtimes; aggregator=Direct())
    joint = merge(merge(sn, rn), xn)

    tp = (first(dtimes), last(dtimes))
    p = vcat(ps, pr, px)
    dprob = DiscreteProblem(joint, copy(u0), tp, p)
    jprob = JumpProblem(joint, dprob, aggregator)

    SRXsystem(sn, rn, xn, u0, ps, pr, px, dtimes, jprob)
end

struct CompiledSRXsystem{JP, XD, JPC, XDC, IXC, DXC}
    system::SRXsystem
    marginal_ensemble::MarginalEnsemble{JP, XD}
    conditional_ensemble::ConditionalEnsemble{JPC, XDC, IXC, DXC}
end

compile(s::SRXsystem) = CompiledSRXsystem(s, MarginalEnsemble(s), ConditionalEnsemble(s))
marginal_density(csrx::CompiledSRXsystem, algorithm, conf::SRXconfiguration) = log_marginal(simulate(algorithm, marginal_configuration(conf), csrx.marginal_ensemble))
conditional_density(csrx::CompiledSRXsystem, algorithm, conf::SRXconfiguration) = log_marginal(simulate(algorithm, conf, csrx.conditional_ensemble))

tspan(sys::JumpNetwork) = (first(sys.dtimes), last(sys.dtimes))

reaction_network(system::SXsystem) = merge(system.sn, system.xn)
reaction_network(system::SRXsystem) = merge(merge(system.sn, system.rn), system.xn)

function _solve(system::SXsystem)
    sol = solve(system.jump_problem, SSAStepper())
end

function generate_configuration(system::SXsystem)
    joint = reaction_network(system)
    integrator = init(system.jump_problem, SSAStepper())
    trajectory = collect_trajectory(SSAIter(integrator))

    s_spec = independent_species(system.sn)
    s_idxs = species_indices(joint, s_spec)
    s_traj = collect_trajectory(sub_trajectory(trajectory, s_idxs))

    x_spec = independent_species(system.xn)
    x_idxs = species_indices(joint, x_spec)
    x_traj = collect_trajectory(sub_trajectory(trajectory, x_idxs))

    SXconfiguration(s_traj, x_traj)
end

function _solve(system::SRXsystem)
    sol = solve(system.jump_problem, SSAStepper())
end

function generate_configuration(system::SRXsystem)
    # we first generate a joint SRX trajectory
    joint = reaction_network(system)
    sol = Trajectory(_solve(system))

    # then we extract the signal
    s_spec = independent_species(system.sn)
    s_idxs = species_indices(joint, s_spec)
    s_traj = collect_trajectory(sub_trajectory(sol, s_idxs))

    # the R trajectory
    r_spec = independent_species(system.rn)
    r_idxs = species_indices(joint, r_spec)
    r_traj = collect_trajectory(sub_trajectory(sol, r_idxs))
    
    # finally we extract the X part from the SRX trajectory
    x_spec = independent_species(system.xn)
    x_idxs = species_indices(joint, x_spec)
    x_traj = collect_trajectory(sub_trajectory(sol, x_idxs))

    SRXconfiguration(s_traj, r_traj, x_traj)
end

function MarginalEnsemble(system::SXsystem)
    joint = reaction_network(system)
    s_idxs = species_indices(joint, Catalyst.species(system.sn))

    dprob = DiscreteProblem(system.sn, system.u0[s_idxs], tspan(system), system.ps)
    jprob = JumpProblem(system.sn, dprob, Direct(), save_positions=(false, false))

    update_map = Int[]
    k = 1
    for react in Catalyst.reactions(joint)
        if react in Catalyst.reactions(system.xn)
            push!(update_map, k)
            k += 1
        else
            push!(update_map, 0)
        end
    end

    MarginalEnsemble(jprob, distribution(joint; update_map), vcat(system.ps, system.px), collect(system.dtimes))
end

function MarginalEnsemble(system::SRXsystem)
    sr_network = merge(system.sn, system.rn)
    joint = merge(sr_network, system.xn)
    sr_idxs = species_indices(joint, Catalyst.species(sr_network)...)

    dprob = DiscreteProblem(sr_network, system.u0[sr_idxs], tspan(system), vcat(system.ps, system.pr))
    # we have to remake the discrete problem with u0 as StaticArray for 
    # improved performance
    # dprob = remake(dprob, u0=SVector(dprob.u0...))
    jprob = JumpProblem(sr_network, dprob, Direct(), save_positions=(false, false))

    MarginalEnsemble(jprob, distribution(system.xn), Int[], system.px, collect(system.dtimes))
end
mutable struct TrajectoryCallback{uType,tType}
    traj::Trajectory{uType,tType}
    index::Int
end

TrajectoryCallback(traj::Trajectory) = TrajectoryCallback(traj, 1)

function (tc::TrajectoryCallback)(integrator::DiffEqBase.DEIntegrator) # affect!
    traj = tc.traj
    cond_u = traj.u[tc.index]
    tc.index = min(tc.index + 1, length(tc.traj.t))
    for i in eachindex(cond_u)
        integrator.u[i] = cond_u[i]
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


function ConditionalEnsemble(
    rn::ReactionSystem, 
    xn::ReactionSystem,
    u0::AbstractVector,
    rp::Vector{Float64},
    xp::Vector{Float64},
    dtimes
)
    dprob = DiscreteProblem(rn, u0, (first(dtimes), last(dtimes)), rp)
    # we have to remake the discrete problem with u0 as StaticArray for 
    # improved performance
    # dprob = remake(dprob, u0=SVector(dprob.u0...))
    jprob = JumpProblem(rn, dprob, Direct(), save_positions=(false, false))

    indep_species = independent_species(rn)
    indep_idxs = species_indices(rn, indep_species)

    dep_species = dependent_species(xn)
    dep_idxs = indexin(species_indices(rn, dep_species), indep_idxs)

    ConditionalEnsemble(jprob, distribution(xn), indep_idxs, dep_idxs, xp, collect(dtimes))
end

function ConditionalEnsemble(system::SRXsystem)
    joint = merge(merge(system.sn, system.rn), system.xn)
    r_idxs = species_indices(joint, Catalyst.species(system.rn))
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

function species_indices(rs::ReactionSystem, species)
    getindex.(Ref(Catalyst.speciesmap(rs)), species)
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

function sample(configuration::T, system::MarginalEnsemble; θ=0.0)::T where T <: SXconfiguration
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

    result = Array{Float64,2}(undef, length(system.dtimes), num_samples)
    for result_col ∈ eachcol(result)
        integrator = DiffEqBase.init(jprob, SSAStepper(), numsteps_hint=0)
        iter = SSAIter(integrator)
        cumulative_logpdf!(result_col, system.x_dist, merge_trajectories(iter, initial.x_traj), system.dtimes, params=system.xp)
    end

    result
end

function propagate(conf::SXconfiguration, ensemble::MarginalEnsemble, u0, tspan::Tuple)
    jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
    integrator = DiffEqBase.init(jprob, SSAStepper(), numsteps_hint=0)

    iter = SSAIter(integrator)
    log_weight = logpdf(ensemble.x_dist, merge_trajectories(iter, conf.x_traj), params=ensemble.xp)

    integrator.u, log_weight
end

function energy_difference(configuration::SXconfiguration, system::MarginalEnsemble)
    -cumulative_logpdf(system.x_dist, merge_trajectories(configuration.s_traj, configuration.x_traj), system.dtimes, params=system.xp)
end

function sample(configuration::T, system::ConditionalEnsemble; θ=0.0)::T where T <: SRXconfiguration
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

    result = Array{Float64,2}(undef, length(system.dtimes), num_samples)
    for result_col ∈ eachcol(result)
        integrator = DiffEqBase.init(jprob, SSAStepper(), callback=cb, tstops=initial.s_traj.t, numsteps_hint=0)
        iter = SSAIter(integrator)
        dep = sub_trajectory(iter, idxs)
        cumulative_logpdf!(result_col, system.x_dist, merge_trajectories(dep, initial.x_traj), system.dtimes, params=system.xp)
    end

    result
end

function simulate(algorithm::DirectMCEstimate, initial::Union{SXconfiguration,SRXconfiguration}, system)
    samples = collect_samples(initial, system, algorithm.num_samples)
    DirectMCResult(samples)
end

function propagate(conf::SRXconfiguration, ensemble::ConditionalEnsemble, u0, tspan::Tuple)
    s_traj = get_slice(conf.s_traj, tspan)
    cb = TrajectoryCallback(s_traj)
    cb = DiscreteCallback(cb, cb, save_positions=(false, false))
    jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
    integrator = DiffEqBase.init(jprob, SSAStepper(), callback=cb, tstops=s_traj.t, numsteps_hint=0)

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
