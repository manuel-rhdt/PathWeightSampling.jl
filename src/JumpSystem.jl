
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

function generate_trace(system::MarkovJumpSystem; u0=system.u0, tspan=system.tspan)
    agg = initialize_aggregator(system.agg, system.reactions, u0=copy(u0), tspan=tspan)

    trace = ReactionTrace([], [])
    while agg.tprev < tspan[2]
        agg = step_ssa(agg, system.reactions, nothing, trace)
    end

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

    while agg.tprev < tspan[2]
        agg = step_ssa(agg, system.reactions, trace, nothing)
    end

    agg.weight
end


struct HybridJumpSystem{A,JS,U,Prob}
    agg::A
    reactions::JS
    u0::U
    tspan::Tuple{Float64,Float64}
    sde_prob::Prob
    sde_dt::Float64
end

function HybridJumpSystem(
    alg::AbstractJumpRateAggregatorAlgorithm,
    reactions::AbstractJumpSet,
    u0::AbstractVector,
    tspan::Tuple{Float64,Float64},
    sde_prob,
    sde_dt,
    ridtogroup,
    traced_reactions::BitSet
)
    agg = build_aggregator(alg, reactions, ridtogroup)
    agg = @set agg.traced_reactions = traced_reactions
    HybridJumpSystem(agg, reactions, u0, tspan, sde_prob, sde_dt)
end

function generate_trace(system::HybridJumpSystem; u0=system.u0, tspan=system.tspan)
    agg = initialize_aggregator(system.agg, system.reactions, u0=copy(u0), tspan=tspan)
    s_prob = remake(system.sde_prob, tspan=tspan, u0=[0.0, u0[1]])

    dt = system.sde_dt
    integrator = init(s_prob, EM(), dt=dt)

    trace = HybridTrace(Float64[], Int16[], Float64[], dt)
    tstop = tspan[1] + dt
    while tstop <= tspan[2]
        agg = set_tspan(agg, (tspan[1], tstop))
        while agg.tprev < tstop
            agg = step_ssa(agg, system.reactions, nothing, trace)
        end
        step!(integrator, dt)
        agg.u[1] = integrator.u[end]
        push!(trace.u, integrator.u[end])
        agg = update_rates(agg, system.reactions)
        tstop += dt
    end

    agg, trace
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
        agg = set_tspan(agg, (tspan[1], tstop))
        while agg.tprev < tstop
            agg = step_ssa(agg, system.reactions, trace, nothing)
        end

        if i <= length(trace.u)
            agg.u[1] = trace.u[i]
            i += 1
            agg = update_rates(agg, system.reactions)
        end
    end

    agg.weight
end

function generate_trajectory(system::Union{MarkovJumpSystem,HybridJumpSystem}, dtimes; u0=system.u0, driving_traj=nothing)
    tspan = extrema(dtimes)
    agg = initialize_aggregator(system.agg, system.reactions, u0=copy(u0), tspan=tspan)

    traj = zeros(eltype(u0), (length(u0), length(dtimes)))
    traj[:, 1] .= u0
    if !isnothing(driving_traj)
        agg.u[1] = driving_traj[1]
    end
    for i in eachindex(dtimes)[2:end]
        agg = set_tspan(agg, (dtimes[1], dtimes[i]))
        while agg.tprev < dtimes[i]
            agg = step_ssa(agg, system.reactions, nothing, nothing)
        end
        traj[:, i] .= agg.u
        if !isnothing(driving_traj)
            agg.u[1] = driving_traj[i]
            agg = update_rates(agg, system.reactions)
        end
    end

    agg, traj
end
