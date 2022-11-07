
struct MarkovJumpSystem{A,R,U}
    agg::A
    reactions::R
    u0::U
    tspan::Tuple{Float64,Float64}
end

function MarkovJumpSystem(
    alg::AbstractJumpRateAggregatorAlgorithm,
    reactions::AbstractJumpSet,
    u0::AbstractVector,
    tspan::Tuple{Float64,Float64},
    ridtogroup,
    traced_reactions::BitSet
)
    agg = build_aggregator(alg, reactions, ridtogroup)
    agg = @set agg.traced_reactions = traced_reactions
    MarkovJumpSystem(agg, reactions, u0, tspan)
end

# to compute the marginal entropy
# 1. simulate input & output and record only output trace
# 2. simulate inputs with output deactivated, average likelihoods

function generate_trace(system::MarkovJumpSystem; u0=system.u0, tspan=system.tspan, seed=nothing)
    agg = initialize_aggregator(system.agg, system.reactions, u0=copy(u0), tspan=tspan, seed=seed)
    trace = ReactionTrace([], [])
    agg = advance_ssa(agg, system.reactions, tspan[2], nothing, trace)
    agg, trace
end

function sample(trace::ReactionTrace, system::MarkovJumpSystem; u0=system.u0, tspan=system.tspan)
    # deactivate all traced reactions
    active_reactions = BitSet(1:length(system.reactions.rates))
    setdiff!(active_reactions, system.agg.traced_reactions)

    agg = initialize_aggregator(
        system.agg,
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
        traced_reactions=BitSet(),
    )

    agg = advance_ssa(agg, system.reactions, tspan[2], trace, nothing)
    agg.weight
end


struct HybridJumpSystem{A,JS,U,Prob}
    agg::A
    reactions::JS
    u0::U
    tspan::Tuple{Float64,Float64}
    dt::Float64
    sde_prob::Prob
    sde_dt::Float64
end

function HybridJumpSystem(
    alg::AbstractJumpRateAggregatorAlgorithm,
    reactions::AbstractJumpSet,
    u0::AbstractVector,
    tspan::Tuple{Float64,Float64},
    dt::Real,
    sde_prob,
    sde_dt::Real,
    ridtogroup,
    traced_reactions::BitSet
)
    agg = build_aggregator(alg, reactions, ridtogroup)
    agg = @set agg.traced_reactions = traced_reactions
    HybridJumpSystem(agg, reactions, u0, tspan, dt, sde_prob, sde_dt)
end

# Advance the aggregator until `t_end`.
function advance_ssa(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet,
    t_end::Float64,
    trace::Union{Nothing,<:ReactionTrace},
    out_trace::Union{Nothing,<:Trace}
)
    tspan = (agg.tprev, t_end)
    agg = set_tspan(agg, tspan)
    while agg.tprev < tspan[2]
        agg = step_ssa(agg, reactions, trace, out_trace)
    end
    agg
end

function advance_ssa(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet,
    t_end::Float64,
    trace::HybridTrace,
    out_trace::Union{Nothing,<:Trace}
)
    dtimes = trace.dtimes
    i1 = searchsortedfirst(dtimes, agg.tprev)
    i2 = searchsortedlast(dtimes, t_end)
    for i in i1:i2
        tstop = dtimes[i]
        agg = advance_ssa(agg, reactions, tstop, ReactionTrace(trace), out_trace)
        agg.u[1] = trace.u[i]
        agg = update_rates(agg, reactions)
    end
    if dtimes[i2] != t_end
        agg = advance_ssa(agg, reactions, t_end, ReactionTrace(trace), out_trace)
    end
    agg
end

function advance_ssa_sde(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet,
    integrator,
    t_end::Float64,
    trace::Union{Nothing,<:ReactionTrace},
    out_trace::Union{Nothing,<:HybridTrace}
)
    add_tstop!(integrator, t_end)
    step!(integrator)
    while integrator.t <= t_end
        agg = advance_ssa(agg, reactions, integrator.t, trace, out_trace)
        agg.u[1] = integrator.u[end]
        if out_trace !== nothing
            push!(out_trace.dtimes, integrator.t)
            push!(out_trace.u, integrator.u[end])
        end
        agg = update_rates(agg, reactions)
        if integrator.t == t_end
            break
        end
        step!(integrator)
    end

    agg
end

function generate_trace(system::HybridJumpSystem; u0=system.u0, tspan=system.tspan, traj=nothing, seed=nothing)
    agg = initialize_aggregator(system.agg, system.reactions, u0=copy(u0), tspan=tspan, seed=seed)
    s_prob = remake(system.sde_prob, tspan=tspan, u0=[0.0, u0[1]])

    if traj !== nothing
        traj[:, 1] .= u0
    end

    dt = system.dt
    sde_dt = system.sde_dt
    if seed === nothing
        integrator = init(s_prob, EM(), dt=sde_dt, save_start=false, save_everystep=false, save_end=false)
    else
        integrator = init(s_prob, EM(), dt=sde_dt, save_start=false, save_everystep=false, save_end=false, seed=seed)
    end

    trace = HybridTrace(Float64[], Int16[], Float64[], Float64[])
    tstops = range(tspan[1], tspan[2], step=dt)
    for (i, tstop) in enumerate(tstops[2:end])
        agg = advance_ssa_sde(agg, system.reactions, integrator, tstop, nothing, trace)
        if traj !== nothing
            traj[:, i+1] .= agg.u
        end
    end

    agg, trace
end

function sample(trace::ReactionTrace, system::HybridJumpSystem; u0=system.u0, tspan=system.tspan)
    # deactivate all traced reactions
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, system.agg.traced_reactions)

    agg = initialize_aggregator(
        system.agg,
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
        traced_reactions=BitSet(),
    )

    dt = system.sde_dt
    s_prob = remake(system.sde_prob, tspan=tspan, u0=[0.0, u0[1]])
    integrator = init(s_prob, EM(), dt=dt / 5, save_start=false, save_everystep=false, save_end=false)

    tstops = range(tspan[1], tspan[2], step=dt)
    for tstop in tstops[2:end]
        agg = advance_ssa_sde(agg, system.reactions, integrator, tstop, trace, nothing)
    end

    agg.weight
end

function sample(trace::HybridTrace, system::HybridJumpSystem; u0=system.u0, tspan=system.tspan)
    # deactivate all traced reactions
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, system.agg.traced_reactions)

    agg = initialize_aggregator(
        system.agg,
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
        traced_reactions=BitSet(),
    )

    dt = trace.dt
    tstops = range(tspan[1], tspan[2], step=dt)
    i = round(Int64, (tspan[1] - system.tspan[1]) / dt) + 1
    for tstop in tstops[2:end]
        agg = advance_ssa(agg, system.reactions, tstop, trace, nothing)
        if i <= length(trace.u)
            agg.u[1] = trace.u[i]
            i += 1
            agg = update_rates(agg, system.reactions)
        end
    end

    agg.weight
end

function generate_trajectory(system::Union{MarkovJumpSystem,HybridJumpSystem}, dtimes; u0=system.u0, driving_traj=nothing, seed=nothing)
    tspan = extrema(dtimes)
    agg = initialize_aggregator(system.agg, system.reactions, u0=copy(u0), tspan=tspan, seed=seed)

    traj = zeros(eltype(u0), (length(u0), length(dtimes)))
    traj[:, 1] .= u0
    if !isnothing(driving_traj)
        agg.u[1] = driving_traj[1]
    end
    for i in eachindex(dtimes)[2:end]
        agg = advance_ssa(agg, system.reactions, dtimes[i], nothing, nothing)
        traj[:, i] .= agg.u
        if !isnothing(driving_traj)
            agg.u[1] = driving_traj[i]
            agg = update_rates(agg, system.reactions)
        end
    end

    agg, traj
end

struct MarkovParticle{Agg}
    agg::Agg
end

struct HybridParticle{Agg,Integrator}
    agg::Agg
    integrator::Integrator
end

weight(particle::MarkovParticle) = particle.agg.weight
weight(particle::HybridParticle) = particle.agg.weight

function MarkovParticle(setup::Setup)
    system = setup.ensemble
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, system.agg.traced_reactions)

    tspan = system.tspan
    u0 = system.u0

    agg = initialize_aggregator(
        copy(system.agg),
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
    )

    MarkovParticle(agg)
end

function HybridParticle(setup::Setup{<:ReactionTrace})
    system = setup.ensemble
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, system.agg.traced_reactions)

    tspan = system.tspan
    u0 = system.u0

    agg = initialize_aggregator(
        copy(system.agg),
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
    )

    s_prob = remake(system.sde_prob, u0=[0.0, u0[1]])
    sde_dt = system.sde_dt
    seed = rand(agg.rng, UInt64)
    integrator = init(s_prob, EM(), dt=sde_dt, save_everystep=false, save_start=false, save_end=false, seed=seed)

    HybridParticle(agg, integrator)
end

function MarkovParticle(parent::MarkovParticle, setup::Setup)
    agg = copy(parent.agg)
    agg = @set agg.weight = 0.0
    MarkovParticle(agg)
end

function HybridParticle(parent::HybridParticle, setup::Setup)
    system = setup.ensemble
    agg = copy(parent.agg)
    agg = @set agg.weight = 0.0

    s_prob = remake(system.sde_prob, u0=copy(parent.integrator.u))
    sde_dt = system.sde_dt
    seed = rand(agg.rng, UInt64)
    integrator = init(s_prob, EM(), dt=sde_dt, save_everystep=false, save_start=false, save_end=false, seed=seed)

    HybridParticle(agg, integrator)
end

function propagate(particle::MarkovParticle, tspan, setup::Setup)
    system = setup.ensemble
    trace = setup.configuration
    agg = particle.agg
    agg = @set agg.weight = 0.0
    agg = advance_ssa(agg, system.reactions, tspan[2], trace, nothing)
    MarkovParticle(agg)
end

function propagate(particle::HybridParticle, tspan, setup::Setup)
    system = setup.ensemble
    trace = setup.configuration
    agg = particle.agg
    agg = @set agg.weight = 0.0
    integrator = particle.integrator
    reinit!(particle.integrator, particle.integrator.u, t0=tspan[1], tf=tspan[2], reinit_cache=false)
    agg = advance_ssa_sde(agg, system.reactions, integrator, tspan[2], trace, nothing)
    HybridParticle(agg, integrator)
end

discrete_times(setup::Setup{<:Trace,<:HybridJumpSystem}) = range(setup.ensemble.tspan[1], setup.ensemble.tspan[2], step=setup.ensemble.dt)

struct TraceAndTrajectory{Trace}
    trace::Trace
    traj::Matrix{Float64}
end
summary(t::TraceAndTrajectory) = t.traj

function generate_configuration(system::HybridJumpSystem; seed=nothing)
    traj = zeros(Float64, (length(system.u0), length(system.tspan[1]:system.dt:system.tspan[2])))
    agg, trace = generate_trace(system; traj=traj, seed=seed)
    TraceAndTrajectory(trace, traj)
end

marginal_density(system::HybridJumpSystem, algorithm::DirectMCEstimate, conf::TraceAndTrajectory; kwargs...) = log_marginal(simulate(SMCEstimate(algorithm.num_samples), ReactionTrace(conf.trace), system; new_particle=HybridParticle, resample_threshold=0, kwargs...))
conditional_density(system::HybridJumpSystem, algorithm::DirectMCEstimate, conf::TraceAndTrajectory; kwargs...) = log_marginal(simulate(SMCEstimate(algorithm.num_samples), conf.trace, system; new_particle=MarkovParticle, resample_threshold=0, kwargs...))

marginal_density(system::HybridJumpSystem, algorithm::SMCEstimate, conf::TraceAndTrajectory; kwargs...) = log_marginal(simulate(algorithm, ReactionTrace(conf.trace), system; new_particle=HybridParticle, kwargs...))
conditional_density(system::HybridJumpSystem, algorithm::SMCEstimate, conf::TraceAndTrajectory; kwargs...) = log_marginal(simulate(algorithm, conf.trace, system; new_particle=MarkovParticle, kwargs...))
compile(system::HybridJumpSystem) = system
