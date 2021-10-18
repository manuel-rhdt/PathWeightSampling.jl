using Transducers
using DiffEqJump

struct MarginalEnsemble{JP <: DiffEqBase.AbstractJumpProblem,U0}
    jump_problem::JP
    dist::TrajectoryDistribution
    dtimes::Vector{Float64}

    # Handling the initial condition. Can be either a vector that specifies the exact initial
    # condition, or an EmpiricalDistribution specifying a probability distribution over initial
    # conditions.
    u0::U0
end

struct ConditionalEnsemble{JP <: DiffEqBase.AbstractJumpProblem,IX,DX}
    jump_problem::JP
    dist::TrajectoryDistribution
    indep_idxs::IX
    dep_idxs::DX
    dtimes::Vector{Float64}
end

struct SXconfiguration{uTs,uTx,tType}
    s_traj::Trajectory{uTs,tType}
    x_traj::Trajectory{uTx,tType}
end

Base.copy(c::SXconfiguration) = SXconfiguration(copy(c.s_traj), copy(c.x_traj))
struct SRXconfiguration{uTs,Utr,Utx,tType}
    s_traj::Trajectory{uTs,tType}
    r_traj::Trajectory{Utr,tType}
    x_traj::Trajectory{Utx,tType}
end

Base.copy(c::SRXconfiguration) = SRXconfiguration(copy(c.s_traj), copy(c.r_traj), copy(c.x_traj))

ensurevec(a::AbstractVector) = a
ensurevec(a) = SVector(a)

function initial_log_likelihood(ensemble::ConditionalEnsemble, u0, x_traj)
    0.0
end

function initial_log_likelihood(ensemble::MarginalEnsemble, u0::AbstractVector, x_traj::Trajectory)
    s0 = u0[1]
    x0 = x_traj.u[1][1]
    if ensemble.u0 isa EmpiricalDistribution
        # compute log P(x0 | s0) = log P(s0, x0) - log ∑ₓ P(s0, x)
        logpdf(ensemble.u0, [s0, x0]) - log(sum(x -> pdf(ensemble.u0, [s0, x]), ensemble.u0.axes[2]))
    else
        0.0
    end
end

function Base.getindex(conf::SXconfiguration, index)
    merge_trajectories(conf.s_traj, conf.x_traj) |> Map((u, t, i)::Tuple -> (ensurevec(u[index]), t, i)) |> Thin() |> collect_trajectory
end

function Base.getindex(conf::SRXconfiguration, index)
    merge_trajectories(conf.s_traj, conf.r_traj, conf.x_traj) |> Map((u, t, i)::Tuple -> (ensurevec(u[index]), t, i)) |> Thin() |> collect_trajectory
end

sample_initial_condition(u0::AbstractVector) = copy(u0)
sample_initial_condition(u0::EmpiricalDistribution) = rand(u0)

sample_initial_condition(ens::MarginalEnsemble) = ens.u0 isa EmpiricalDistribution ? rand(ens.u0)[[1]] : ens.u0
sample_initial_condition(ens::ConditionalEnsemble) = ens.jump_problem.prob.u0

abstract type JumpNetwork end

"""
    SimpleSystem(input_network, output_network, u0, ps, px, dtimes, dist=nothing)

A `SimpleSystem` consists of two ModelingToolkit `ReactionSystem`s:
One that generates the input trajectories, and one that generates the
output trajectories.
"""
struct SimpleSystem <: JumpNetwork
    sn::ReactionSystem
    xn::ReactionSystem

    u0::Union{AbstractVector,EmpiricalDistribution}

    ps::AbstractVector
    px::AbstractVector

    dtimes

    jump_problem::AbstractJumpProblem
    dist::TrajectoryDistribution
end

function SimpleSystem(sn, xn, u0, ps, px, dtimes, dist=nothing)
    joint = merge(sn, xn)

    tp = (first(dtimes), last(dtimes))
    p = vcat(ps, px)
    dprob = DiscreteProblem(joint, sample_initial_condition(u0), tp, p)
    jprob = JumpProblem(convert(ModelingToolkit.JumpSystem, joint), dprob, Direct(), save_positions=(false, false))

    if dist === nothing
        update_map = build_update_map(joint, xn)
        dist = distribution(joint, p; update_map)
    end

    SimpleSystem(sn, xn, u0, ps, px, dtimes, jprob, dist)
end

struct CompiledSimpleSystem{JP}
    system::SimpleSystem
    marginal_ensemble::MarginalEnsemble{JP}
end

compile(s::SimpleSystem) = CompiledSimpleSystem(s, MarginalEnsemble(s))
marginal_density(csx::CompiledSimpleSystem, algorithm, conf::SXconfiguration) = log_marginal(simulate(algorithm, conf, csx.marginal_ensemble))
conditional_density(csx::CompiledSimpleSystem, algorithm, conf::SXconfiguration) = -energy_difference(conf, csx.marginal_ensemble)

"""
    ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes, dist=nothing; aggregator=Direct())

A `ComplexSystem` consists of three ModelingToolkit `ReactionSystem`s:
One that generates the input trajectories, and one that generates the
output trajectories.
"""
struct ComplexSystem <: JumpNetwork
    sn::ReactionSystem
    rn::ReactionSystem
    xn::ReactionSystem

    u0::Union{AbstractVector,EmpiricalDistribution}

    ps::AbstractVector
    pr::AbstractVector
    px::AbstractVector

    dtimes

    jump_problem
    dist::TrajectoryDistribution
end

function ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes, dist=nothing; aggregator=Direct())
    joint = merge(merge(sn, rn), xn)

    tp = (first(dtimes), last(dtimes))
    p = vcat(ps, pr, px)
    dprob = DiscreteProblem(joint, sample_initial_condition(u0), tp, p)
    jprob = JumpProblem(convert(ModelingToolkit.JumpSystem, joint), dprob, aggregator, save_positions=(false, false))

    if dist === nothing
        update_map = build_update_map(joint, xn)
        dist = distribution(joint, p; update_map)
    end

    ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes, jprob, dist)
end

struct CompiledComplexSystem{JP,JPC,IXC,DXC}
    system::ComplexSystem
    marginal_ensemble::MarginalEnsemble{JP}
    conditional_ensemble::ConditionalEnsemble{JPC,IXC,DXC}
end

compile(s::ComplexSystem) = CompiledComplexSystem(s, MarginalEnsemble(s), ConditionalEnsemble(s))
marginal_density(csrx::CompiledComplexSystem, algorithm, conf::SRXconfiguration) = log_marginal(simulate(algorithm, marginal_configuration(conf), csrx.marginal_ensemble))
conditional_density(csrx::CompiledComplexSystem, algorithm, conf::SRXconfiguration) = log_marginal(simulate(algorithm, conf, csrx.conditional_ensemble))

tspan(sys::JumpNetwork) = (first(sys.dtimes), last(sys.dtimes))

reaction_network(system::SimpleSystem) = merge(system.sn, system.xn)
reaction_network(system::ComplexSystem) = merge(merge(system.sn, system.rn), system.xn)

function _solve(system::SimpleSystem)
    sol = solve(system.jump_problem, SSAStepper())
end

function generate_configuration(system::SimpleSystem)
    joint = reaction_network(system)
    jp = remake(system.jump_problem, u0=sample_initial_condition(system.u0))
    integrator = init(jp, SSAStepper(), tstops=())
    trajectory_iter = SSAIter(integrator)
    trajectory = trajectory_iter |> collect_trajectory


    s_spec = independent_species(system.sn)
    s_idxs = sort(SVector(species_indices(joint, s_spec)...))
    s_traj = sub_trajectory(trajectory, s_idxs)

    x_spec = independent_species(system.xn)
    x_idxs = sort(SVector(species_indices(joint, x_spec)...))
    x_traj = sub_trajectory(trajectory, x_idxs)

    SXconfiguration(s_traj, x_traj)
end

function _solve(system::ComplexSystem)
    sol = solve(system.jump_problem, SSAStepper())
end

function generate_configuration(system::ComplexSystem)
    # we first generate a joint SRX trajectory
    joint = reaction_network(system)
    jp = remake(system.jump_problem, u0=sample_initial_condition(system.u0))
    integrator = init(jp, SSAStepper(), tstops=())
    trajectory = SSAIter(integrator) |> collect_trajectory

    # then we extract the signal
    s_spec = independent_species(system.sn)
    s_idxs = sort(SVector(species_indices(joint, s_spec)...))
    s_traj = sub_trajectory(trajectory, s_idxs)

    # the R trajectory
    r_spec = independent_species(system.rn)
    r_idxs = sort(SVector(species_indices(joint, r_spec)...))
    r_traj = sub_trajectory(trajectory, r_idxs)
    
    # finally we extract the X part from the SRX trajectory
    x_spec = independent_species(system.xn)
    x_idxs = sort(SVector(species_indices(joint, x_spec)...))
    x_traj = sub_trajectory(trajectory, x_idxs)

    SRXconfiguration(s_traj, r_traj, x_traj)
end

function build_update_map(joint::ReactionSystem, xn::ReactionSystem)
    update_map = Int[]
    spmap = Catalyst.speciesmap(joint)
    mapper = (x, y)::Pair -> spmap[x] => y

    unique_netstoich = unique(map(r -> mapper.(r.netstoich), Catalyst.reactions(xn)))

    for react in Catalyst.reactions(joint)
        new_index = 0
        for (k, un) in enumerate(unique_netstoich)
            if mapper.(react.netstoich) == un
                new_index = k
                break
            end
        end

        push!(update_map, new_index)
    end
    update_map
end

function MarginalEnsemble(system::SimpleSystem)
    joint = reaction_network(system)
    s_idxs = species_indices(joint, Catalyst.species(system.sn))

    dprob = DiscreteProblem(system.sn, sample_initial_condition(system.u0)[s_idxs], tspan(system), system.ps)
    jprob = JumpProblem(convert(ModelingToolkit.JumpSystem, system.sn), dprob, Direct(), save_positions=(false, false))

    u0 = system.u0 isa EmpiricalDistribution ? system.u0 : system.u0[s_idxs]
    MarginalEnsemble(jprob, system.dist, collect(system.dtimes), u0)
end

function MarginalEnsemble(system::ComplexSystem)
    sr_network = merge(system.sn, system.rn)
    joint = merge(sr_network, system.xn)
    sr_idxs = species_indices(joint, Catalyst.species(sr_network))

    dprob = DiscreteProblem(sr_network, sample_initial_condition(system.u0)[sr_idxs], tspan(system), vcat(system.ps, system.pr))
    jprob = JumpProblem(convert(ModelingToolkit.JumpSystem, sr_network), dprob, Direct(), save_positions=(false, false))

    MarginalEnsemble(jprob, system.dist, collect(system.dtimes), system.u0[sr_idxs])
end


function ConditionalEnsemble(system::ComplexSystem)
    joint = merge(merge(system.sn, system.rn), system.xn)
    r_idxs = species_indices(joint, Catalyst.species(system.rn))
    dprob = DiscreteProblem(system.rn, sample_initial_condition(system.u0)[r_idxs], (first(system.dtimes), last(system.dtimes)), system.pr)
    jprob = JumpProblem(convert(ModelingToolkit.JumpSystem, system.rn), dprob, Direct(), save_positions=(false, false))

    indep_species = independent_species(system.rn)
    indep_idxs = species_indices(system.rn, indep_species)

    dep_species = dependent_species(system.xn)
    dep_idxs = indexin(species_indices(system.rn, dep_species), indep_idxs)

    ConditionalEnsemble(jprob, system.dist, indep_idxs, dep_idxs, collect(system.dtimes))
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
    jprob = remake(system.jump_problem, u0=sample_initial_condition(system))
    integrator = DiffEqBase.init(jprob, SSAStepper(), tstops=(), numsteps_hint=0)
    iter = SSAIter(integrator)
    s_traj = collect_trajectory(iter)
    SXconfiguration(s_traj, configuration.x_traj)
end

function collect_samples(initial::SXconfiguration, ensemble::MarginalEnsemble, num_samples::Int)
    result = Array{Float64,2}(undef, length(ensemble.dtimes), num_samples)
    for result_col ∈ eachcol(result)
        u0 = sample_initial_condition(ensemble)
        jprob = remake(ensemble.jump_problem, u0=u0)
        integrator = DiffEqBase.init(jprob, SSAStepper(), tstops=(), numsteps_hint=0)
        iter = SSAIter(integrator) |> Map((u, t, i)::Tuple -> (u, t, 0))
        cumulative_logpdf!(result_col, ensemble.dist, merge_trajectories(iter, initial.x_traj), ensemble.dtimes)
        result_col .+= initial_log_likelihood(ensemble, u0, initial.x_traj)
    end

    result
end

function propagate(conf::SXconfiguration, ensemble::MarginalEnsemble, u0, tspan::Tuple)
    jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
    integrator = DiffEqBase.init(jprob, SSAStepper(), tstops=(), numsteps_hint=0)
    ix1 = max(searchsortedfirst(conf.x_traj.t, tspan[1]) - 1, 1)
    iter = SSAIter(integrator) |> Map((u, t, i)::Tuple -> (u, t, 0))

    log_weight = trajectory_energy(ensemble.dist, iter |> MergeWith(conf.x_traj, ix1), tspan=tspan)

    copy(integrator.u), log_weight
end

function energy_difference(configuration::SXconfiguration, ensemble::MarginalEnsemble)
    log_p0 = initial_log_likelihood(ensemble, configuration.s_traj.u[1], configuration.x_traj)
    - cumulative_logpdf(ensemble.dist, configuration.s_traj |> MergeWith(configuration.x_traj), ensemble.dtimes) .- log_p0
end

function sample(configuration::SRXconfiguration, system::ConditionalEnsemble; θ=0.0)
    if θ != 0.0
        error("can only use DirectMC with JumpNetwork")
    end
    driven_jp = DrivenJumpProblem(system.jump_problem, configuration.s_traj)
    integrator = init(driven_jp)
    iter = SSAIter(integrator)
    rtraj = sub_trajectory(iter, system.indep_idxs)
    SRXconfiguration(configuration.s_traj, rtraj, configuration.x_traj)
end

function collect_samples(initial::SRXconfiguration, ensemble::ConditionalEnsemble, num_samples::Int)
    driven_jp = DrivenJumpProblem(ensemble.jump_problem, initial.s_traj)

    result = Array{Float64,2}(undef, length(ensemble.dtimes), num_samples)
    for result_col ∈ eachcol(result)
        integrator = init(driven_jp)
        iter = SSAIter(integrator) |> Map((u, t, i)::Tuple -> (u, t, 0))
        cumulative_logpdf!(result_col, ensemble.dist, merge_trajectories(iter, initial.x_traj), ensemble.dtimes)
    end

    result
end

function simulate(algorithm::DirectMCEstimate, initial::Union{SXconfiguration,SRXconfiguration}, system)
    samples = collect_samples(initial, system, algorithm.num_samples)
    DirectMCResult(samples)
end

function create_integrator(conf::SRXconfiguration, ensemble::ConditionalEnsemble, u0, tspan::Tuple)
    jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
    driven_jp = DrivenJumpProblem(jprob, conf.s_traj)
    init(driven_jp)
end

function propagate(conf::SRXconfiguration, ensemble::ConditionalEnsemble, u0, tspan::Tuple)
    integrator = create_integrator(conf, ensemble, u0, tspan)
    iter = SSAIter(integrator) |> Map((u, t, i)::Tuple -> (u, t, 0))
    ix1 = max(searchsortedfirst(conf.x_traj.t, tspan[1]), 1)

    log_weight = trajectory_energy(ensemble.dist, iter |> MergeWith(conf.x_traj, ix1), tspan=tspan)

    copy(integrator.u), log_weight
end

function energy_difference(configuration::SRXconfiguration, ensemble::ConditionalEnsemble)
    traj = merge_trajectories(configuration.s_traj, configuration.r_traj, configuration.x_traj)
    -cumulative_logpdf(ensemble.dist, traj, ensemble.dtimes)
end 

function marginal_configuration(conf::SRXconfiguration)
    new_s = conf.s_traj |> MergeWith(conf.r_traj) |> Map((u, t, i)::Tuple -> (SVector(u...), t, i)) |> collect_trajectory
    SXconfiguration(new_s, conf.x_traj)
end

# MCMC Moves in trajectory space

mutable struct TrajectoryChain{Ensemble} <: MarkovChain
    ensemble::Ensemble
    # interaction parameter
    θ::Float64

    # to save statistics
    last_regrowth::Float64
    accepted_list::Vector{Float64}
    rejected_list::Vector{Float64}
end

chain(ensemble; θ::Real=1.0) = TrajectoryChain(ensemble, θ, 0.0, Float64[], Float64[])

# reset statistics
function reset(pot::TrajectoryChain)
    resize!(pot.accepted_list, 0)
    resize!(pot.rejected_list, 0)
end

function accept(pot::TrajectoryChain)
    push!(pot.accepted_list, pot.last_regrowth)
end

function reject(pot::TrajectoryChain)
    push!(pot.rejected_list, pot.last_regrowth)
end

function energy(conf::SXconfiguration, chain::TrajectoryChain; θ=chain.θ) 
    if θ > zero(θ)
        -θ * trajectory_energy(chain.ensemble.dist, conf.s_traj |> MergeWith(conf.x_traj))
    else
        0.0
    end
end

function energy(conf::SRXconfiguration, chain::TrajectoryChain; θ=chain.θ)
    if θ > zero(θ)
        traj = merge_trajectories(conf.s_traj, conf.r_traj, conf.x_traj)
        θ * logpdf(chain.ensemble.dist, traj)
    else
        0.0
    end
end

function propose!(new_conf, old_conf, chain::TrajectoryChain) 
    chain.last_regrowth = propose!(new_conf, old_conf, chain.ensemble)
    new_conf
end
propose!(new_conf::SXconfiguration, old_conf::SXconfiguration, ensemble::MarginalEnsemble) = propose!(new_conf.s_traj, old_conf.s_traj, ensemble)

function propose!(new_traj::Trajectory, old_traj::Trajectory, ensemble::MarginalEnsemble)
    jump_problem = ensemble.jump_problem

    regrow_duration = rand() * duration(old_traj)

    if rand(Bool)
        shoot_forward!(new_traj, old_traj, jump_problem, old_traj.t[end] - regrow_duration)
    else
        shoot_backward!(new_traj, old_traj, jump_problem, old_traj.t[begin] + regrow_duration)
    end

    regrow_duration
end

function shoot_forward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::DiffEqBase.AbstractJumpProblem, branch_time::Real)
    branch_value = old_traj(branch_time)
    branch_point = searchsortedfirst(old_traj.t, branch_time)
    tspan = (branch_time, old_traj.t[end])

    empty!(new_traj.u)
    empty!(new_traj.t)
    empty!(new_traj.i)
    append!(new_traj.u, @view old_traj.u[begin:branch_point - 1])
    append!(new_traj.t, @view old_traj.t[begin:branch_point - 1])
    append!(new_traj.i, @view old_traj.i[begin:branch_point - 1])

    jump_problem = DiffEqBase.remake(jump_problem; u0=branch_value, tspan=tspan)
    integrator = DiffEqBase.init(jump_problem, SSAStepper())
    iter = SSAIter(integrator)
    new_branch = collect_trajectory(iter)

    append!(new_traj.t, new_branch.t)
    append!(new_traj.u, new_branch.u)
    append!(new_traj.i, new_branch.i)
    nothing
end

function shoot_backward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::DiffEqBase.AbstractJumpProblem, branch_time::Real)
    branch_value = old_traj(branch_time)
    branch_point = searchsortedfirst(old_traj.t, branch_time)
    tspan = (old_traj.t[begin], branch_time)

    jump_problem = DiffEqBase.remake(jump_problem; u0=branch_value, tspan=tspan)
    integrator = DiffEqBase.init(jump_problem, SSAStepper())
    iter = SSAIter(integrator)
    new_branch = collect_trajectory(iter)

    empty!(new_traj.u)
    empty!(new_traj.t)
    empty!(new_traj.i)

    append!(new_traj.u, @view new_branch.u[end - 1:-1:begin])
    append!(new_traj.u, @view old_traj.u[branch_point:end])
    append!(new_traj.i, @view new_branch.i[end - 1:-1:begin])
    append!(new_traj.i, @view old_traj.i[branch_point:end])

    for rtime in @view new_branch.t[end:-1:begin + 1]
        push!(new_traj.t, branch_time - rtime)
    end
    append!(new_traj.t, @view old_traj.t[branch_point:end])
    nothing
end