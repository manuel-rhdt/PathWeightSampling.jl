module JumpSystem

export MarkovJumpSystem, HybridJumpSystem

import ..PathWeightSampling: AbstractSystem
import ..PathWeightSampling as PWS
import ..DirectMC: DirectMCEstimate
import ..SMC
using ..SSA
using Accessors
using StochasticDiffEq

import DataFrames: DataFrame
import SciMLBase
using RecipesBase
import StaticArrays
import StaticArrays: StaticArray, setindex

import Random

struct TraceAndTrajectory{Trace,tType,uType}
    trace::Trace
    discrete_times::Vector{tType}
    traj::Matrix{uType}
    species::Vector{Symbol}
end

# handy for quick plotting of configurations
@recipe function f(conf::TraceAndTrajectory{Trace}) where {Trace}
    label --> hcat([String(sym) for sym in conf.species]...)
    conf.discrete_times, conf.traj'
end

function PWS.to_dataframe(trace::TraceAndTrajectory)
    cols = Any[:time=>trace.discrete_times]
    for (species, traj) in zip(trace.species, eachrow(trace.traj))
        push!(cols, species => traj)
    end
    DataFrame(cols)
end

struct MarkovJumpSystem{A,R,U} <: AbstractSystem
    agg::A
    reactions::R
    u0::U
    input_reactions::BitSet
    output_reactions::BitSet
    tspan::Tuple{Float64,Float64}
    dt::Float64
end

function Base.copy(js::MarkovJumpSystem)
    MarkovJumpSystem(
        copy(js.agg),
        js.reactions,
        js.u0,
        js.input_reactions,
        js.output_reactions,
        js.tspan,
        js.dt
    )
end

function MarkovJumpSystem(
    alg::AbstractJumpRateAggregatorAlgorithm,
    reactions::AbstractJumpSet,
    u0::AbstractVector,
    tspan::Tuple{Float64,Float64},
    input_species::Symbol,
    output_species::Symbol,
    dt=(tspan[2] - tspan[1]) / 100
)
    ridtogroup = make_reaction_groups(reactions, output_species)
    input_reactions = reactions_that_mutate_species(reactions, input_species)
    output_reactions = reactions_that_mutate_species(reactions, output_species)
    if !isempty(intersect(input_reactions, output_reactions))
        error("Reactions that directly affect both input and output species are not supported.")
    end
    agg = build_aggregator(alg, reactions, u0, ridtogroup)
    MarkovJumpSystem(agg, reactions, u0, input_reactions, output_reactions, tspan, dt)
end

function generate_trace(system::MarkovJumpSystem; u0=system.u0, tspan=system.tspan, traj=nothing, rng=Random.default_rng())
    agg = system.agg
    all = BitSet(1:SSA.num_reactions(system.reactions))
    initialize_aggregator!(
        agg, 
        system.reactions; 
        u0=copy(u0), 
        traced_reactions=all, 
        active_reactions=all, 
        tspan=tspan, 
        seed=rand(rng, UInt32)
    )

    reaction_times = Float64[]
    reaction_indices = Int16[]
    sizehint!(reaction_times, 100_000)
    sizehint!(reaction_indices, 100_000)

    trace = ReactionTrace(reaction_times, reaction_indices, agg.traced_reactions)

    if traj !== nothing
        traj[:, 1] = u0
    end
    tstops = range(tspan[1], tspan[2], step=system.dt)
    for (i, tstop) in enumerate(tstops[2:end])
        advance_ssa!(agg, system.reactions, tstop, nothing, trace)
        if traj !== nothing
            traj[:, i+1] .= agg.u
        end
    end

    agg, trace
end

function PWS.generate_configuration(system::MarkovJumpSystem; rng=Random.default_rng())
    dtimes = PWS.discrete_times(system)
    traj = zeros(eltype(system.u0), (length(system.u0), length(dtimes)))
    agg, trace = generate_trace(system; traj=traj, rng=rng)
    TraceAndTrajectory(trace, Vector(dtimes), traj, PWS.SSA.speciesnames(system.reactions))
end

function log_probability(system::MarkovJumpSystem, trace::ReactionTrace; u0=system.u0, dtimes=PWS.discrete_times(system), rng=Random.default_rng())
    tspan = extrema(dtimes)

    # deactivate all traced reactions
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, trace.traced_reactions)

    agg = system.agg
    initialize_aggregator!(
        agg,
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
        traced_reactions=BitSet(),
        seed=rand(rng, UInt)
    )
    agg.trace_index = searchsortedfirst(trace.t, tspan[1])

    cond_prob = zeros(Float64, length(dtimes))
    for (i, t) in Iterators.drop(enumerate(dtimes), 1)
        advance_ssa!(agg, system.reactions, t, trace, nothing)
        cond_prob[i] = agg.weight
    end

    cond_prob
end

function PWS.conditional_density(system::MarkovJumpSystem, algorithm, conf::TraceAndTrajectory; full_result=false, kwargs...)
    trace = conf.trace
    traced_reactions = union(system.input_reactions, system.output_reactions)
    trace = filter_trace(trace, traced_reactions)
    if 1:num_reactions(system.reactions) ⊆ traced_reactions
        # we don't need to marginalize
        log_probability(system, trace; kwargs...)
    else
        marginalization_result = PWS.simulate(algorithm, trace, system; Particle=MarkovParticle, kwargs...)
        if full_result
            marginalization_result
        else
            PWS.log_marginal(marginalization_result)
        end
    end
end

function PWS.marginal_density(system::MarkovJumpSystem, algorithm, conf::TraceAndTrajectory; full_result=false, kwargs...)
    trace = conf.trace
    traced_reactions = system.output_reactions
    trace = filter_trace(trace, traced_reactions)
    marginalization_result = PWS.simulate(algorithm, trace, system; Particle=MarkovParticle, kwargs...)
    if full_result
        marginalization_result
    else
        PWS.log_marginal(marginalization_result)
    end
end

struct HybridJumpSystem{A,JS,U,Prob} <: AbstractSystem
    agg::A
    reactions::JS
    u0::U
    tspan::Tuple{Float64,Float64}
    dt::Float64
    sde_prob::Prob
    sde_dt::Float64
    output_reactions::BitSet
    sde_species_mapping::Vector{Pair{Int,Int}}
end

function HybridJumpSystem(
    alg::AbstractJumpRateAggregatorAlgorithm,
    reactions::AbstractJumpSet,
    u0::AbstractVector,
    tspan::Tuple{Float64,Float64},
    dt::Real,
    sde_prob::SciMLBase.SDEProblem,
    sde_dt::Real,
    input_species::Symbol,
    output_species::Symbol,
    sde_species_mapping::Vector{Pair{Int,Int}}
)
    ridtogroup = make_reaction_groups(reactions, output_species)
    input_reactions = reactions_that_mutate_species(reactions, input_species)
    @assert input_reactions == Set() "we do not support reactions that mutate the input"
    output_reactions = reactions_that_mutate_species(reactions, output_species)
    agg = build_aggregator(alg, reactions, u0, ridtogroup)
    HybridJumpSystem(agg, reactions, u0, tspan, dt, sde_prob, sde_dt, output_reactions, sde_species_mapping)
end

function Base.copy(js::HybridJumpSystem)
    HybridJumpSystem(
        copy(js.agg),
        js.reactions,
        js.u0,
        js.tspan,
        js.dt,
        remake(js.sde_prob),
        js.sde_dt,
        js.output_reactions,
        js.sde_species_mapping
    )
end

function Base.show(io::IO, ::MIME"text/plain", system::Union{MarkovJumpSystem, HybridJumpSystem})
    if system isa MarkovJumpSystem
        name = "MarkovJumpSystem"
    else
        name = "HybridJumpSystem"
    end
    println(io, "$name with ", num_species(system.reactions), " species and ", num_reactions(system.reactions), " reactions")
    
    show(io, "text/plain", system.reactions)

    print(io, "\n\nInitial condition:")
    jvars = SSA.speciesnames(system.reactions)
    for i in eachindex(system.u0)
        print(io, "\n    ", jvars[i], " = ", system.u0[i])
    end
end

# Advance the aggregator until `t_end`.
function advance_ssa!(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet,
    t_end::Float64,
    trace::Union{Nothing,<:ReactionTrace},
    out_trace::Union{Nothing,<:Trace}
)
    tspan = (agg.tprev, t_end)
    agg.tspan = tspan
    while agg.tprev < tspan[2]
        step_ssa!(agg, reactions, trace, out_trace)
    end
    # here we know agg.tprev == tspan[2]
    agg
end

@inline function update_ssa_from_sde!(agg::AbstractJumpRateAggregator, sde_u::AbstractVector, sde_species_mapping)
    agg_u = agg.u
    if agg_u isa StaticArray
        for (sde_index, ssa_index) in sde_species_mapping
            agg_u = setindex(agg_u, sde_u[sde_index], ssa_index)
        end
        agg.u = agg_u
    else
        for (sde_index, ssa_index) in sde_species_mapping
            agg_u[ssa_index] = sde_u[sde_index]
        end
    end
end

function advance_ssa!(
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
        advance_ssa!(agg, reactions, tstop, ReactionTrace(trace), out_trace)
        update_ssa_from_sde!(agg, trace.u[i], trace.sde_species_mapping)
        SSA.update_rates!(agg, reactions)
    end
    if agg.tprev < t_end
        advance_ssa!(agg, reactions, t_end, ReactionTrace(trace), out_trace)
    end
    agg
end

function advance_ssa_sde!(
    agg::AbstractJumpRateAggregator,
    reactions::AbstractJumpSet,
    integrator::StochasticDiffEq.SDEIntegrator,
    sde_species_mapping::Vector{Pair{Int,Int}},
    t_end::Float64,
    trace::Union{Nothing,<:ReactionTrace},
    out_trace::Union{Nothing,<:HybridTrace}
)
    add_tstop!(integrator, t_end)
    step!(integrator)
    while integrator.t <= t_end
        advance_ssa!(agg, reactions, integrator.t, trace, out_trace)
        update_ssa_from_sde!(agg, integrator.u, sde_species_mapping)
        SSA.update_rates!(agg, reactions)
        if out_trace !== nothing
            push!(out_trace.dtimes, integrator.t)
            push!(out_trace.u, copy(integrator.u))
        end
        if integrator.t == t_end
            break
        end
        step!(integrator)
    end

    agg
end

function generate_trace(system::HybridJumpSystem; u0=system.u0, tspan=system.tspan, traj=nothing, rng=Random.default_rng())
    all = BitSet(1:SSA.num_reactions(system.reactions))
    agg = initialize_aggregator!(
        system.agg, 
        system.reactions,
        u0=copy(u0),
        traced_reactions=all,
        active_reactions=all,
        tspan=tspan, 
        seed=rand(rng, UInt32)
    )
    s_prob = remake(system.sde_prob, tspan=tspan)

    if traj !== nothing
        traj[:, 1] .= u0
    end

    dt = system.dt
    sde_dt = system.sde_dt
    seed = rand(rng, UInt)
    integrator = init(s_prob, EM(), dt=sde_dt, save_start=false, save_everystep=false, save_end=false, seed=seed)

    reaction_times = Float64[]
    reaction_indices = Int16[]
    sde_u = typeof(s_prob.u0)[]
    sde_t = Float64[]
    sizehint!(reaction_times, 100_000)
    sizehint!(reaction_indices, 100_000)
    sizehint!(sde_u, Int((tspan[2] - tspan[1]) ÷ sde_dt))
    sizehint!(sde_t, Int((tspan[2] - tspan[1]) ÷ sde_dt))

    trace = HybridTrace(reaction_times, reaction_indices, agg.traced_reactions, sde_u, sde_t, system.sde_species_mapping)

    tstops = range(tspan[1], tspan[2], step=dt)
    for (i, tstop) in enumerate(tstops[2:end])
        advance_ssa_sde!(agg, system.reactions, integrator, system.sde_species_mapping, tstop, nothing, trace)
        if traj !== nothing
            traj[:, i+1] .= agg.u
        end
    end

    agg, trace
end

function log_probability(system::HybridJumpSystem, trace::ReactionTrace; u0=system.u0, dtimes=PWS.discrete_times(system), rng=Random.default_rng())
    tspan = extrema(dtimes)

    # deactivate all traced reactions
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, trace.traced_reactions)

    agg = initialize_aggregator!(
        system.agg,
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
        traced_reactions=BitSet(),
        seed=rand(rng, UInt)
    )
    agg.trace_index = searchsortedfirst(trace.t, tspan[1])

    dt = system.sde_dt
    s_prob = remake(system.sde_prob, tspan=tspan)
    seed = rand(rng, UInt)
    integrator = init(s_prob, EM(), dt=dt, save_start=false, save_everystep=false, save_end=false, seed=seed)

    cond_prob = zeros(Float64, length(dtimes))
    for (i, t) in Iterators.drop(enumerate(dtimes), 1)
        advance_ssa_sde!(agg, system.reactions, integrator, system.sde_species_mapping, t, trace, nothing)
        cond_prob[i] = agg.weight
    end

    cond_prob
end

function log_probability(system::Union{MarkovJumpSystem, HybridJumpSystem}, trace::HybridTrace; u0=system.u0, dtimes=PWS.discrete_times(system), rng=Random.default_rng())
    tspan = extrema(dtimes)

    # deactivate all traced reactions
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, trace.traced_reactions)

    agg = initialize_aggregator!(
        system.agg,
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
        traced_reactions=BitSet(),
        seed=rand(rng, UInt)
    )
    agg.trace_index = searchsortedfirst(trace.t, tspan[1])

    cond_prob = zeros(Float64, length(dtimes))
    for (i, t) in Iterators.drop(enumerate(dtimes), 1)
        agg = advance_ssa!(agg, system.reactions, t, trace, nothing)
        cond_prob[i] = agg.weight
    end

    cond_prob
end

struct MarkovParticle{Agg} <: SMC.AbstractParticle
    agg::Agg
end

struct HybridParticle{Agg,Integrator} <: SMC.AbstractParticle
    agg::Agg
    integrator::Integrator
end

SMC.weight(particle::Union{MarkovParticle, HybridParticle}) = particle.agg.weight

function SMC.spawn(::Type{<:MarkovParticle}, setup::SMC.Setup)
    system = setup.ensemble

    # only fire reactions that are not included in the trace
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, setup.configuration.traced_reactions)

    tspan = system.tspan
    u0 = system.u0

    agg = copy(system.agg)
    initialize_aggregator!(
        agg,
        system.reactions;
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
        traced_reactions=BitSet(),
        seed=rand(setup.rng, UInt32)
    )

    MarkovParticle(agg)
end

function SMC.spawn(::Type{<:HybridParticle}, setup::SMC.Setup{<:ReactionTrace})
    system = setup.ensemble
    active_reactions = BitSet(1:num_reactions(system.reactions))
    setdiff!(active_reactions, setup.configuration.traced_reactions)

    tspan = system.tspan
    u0 = system.u0

    agg = copy(system.agg)
    initialize_aggregator!(
        agg,
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        active_reactions=active_reactions,
        traced_reactions=BitSet(),
        seed=rand(setup.rng, UInt32)
    )

    s_prob = remake(system.sde_prob; u0=copy(system.sde_prob.u0))
    sde_dt = system.sde_dt
    seed = rand(setup.rng, UInt)
    integrator = init(s_prob, EM(), dt=sde_dt, save_everystep=false, save_start=false, save_end=false, seed=seed)

    HybridParticle(agg, integrator)
end

function SMC.clone_from!(child::MarkovParticle, parent::MarkovParticle, setup::SMC.Setup)
    if child.agg.u isa StaticArrays.StaticArray
        child.agg.u = copy(parent.agg.u)
    else
        child.agg.u .= parent.agg.u
    end
    child.agg.tstop = parent.agg.tstop
    child.agg.sumrate = parent.agg.sumrate
    child.agg.gsumrate = parent.agg.gsumrate
    child.agg.rates .= parent.agg.rates
    child.agg.grates .= parent.agg.grates
    child.agg.trace_index = parent.agg.trace_index
    child.agg.weight = parent.agg.weight
    if !isnothing(parent.agg.cache)
        child.agg.cache = copy(parent.agg.cache)
    end

    child
end

function SMC.clone_from!(child::HybridParticle, parent::HybridParticle, setup::SMC.Setup)
    if child.agg.u isa StaticArrays.StaticArray
        child.agg.u = copy(parent.agg.u)
    else
        child.agg.u .= parent.agg.u
    end
    child.agg.tstop = parent.agg.tstop
    child.agg.sumrate = parent.agg.sumrate
    child.agg.gsumrate = parent.agg.gsumrate
    child.agg.rates .= parent.agg.rates
    child.agg.grates .= parent.agg.grates
    child.agg.trace_index = parent.agg.trace_index
    child.agg.weight = parent.agg.weight
    if !isnothing(parent.agg.cache)
        child.agg.cache = copy(parent.agg.cache)
    end
    
    reinit!(child.integrator, copy(parent.integrator.u))
    child
end

function SMC.propagate!(particle::MarkovParticle, tspan::Tuple{T,T}, setup::SMC.Setup) where {T<:Real}
    system = setup.ensemble
    trace = setup.configuration
    agg = particle.agg
    agg.weight = 0.0
    agg.tprev = tspan[1]
    agg.tspan = tspan
    # @assert agg.trace_index == searchsortedfirst(trace.t, tspan[1])
    agg.trace_index = searchsortedfirst(trace.t, tspan[1])
    advance_ssa!(agg, system.reactions, tspan[2], trace, nothing)
    particle
end

function SMC.propagate!(particle::HybridParticle, tspan::Tuple{T,T}, setup::SMC.Setup) where {T<:Real}
    system = setup.ensemble
    trace = setup.configuration
    agg = particle.agg
    agg.weight = 0.0
    agg.tprev = tspan[1]
    agg.tspan = tspan
    agg.trace_index = searchsortedfirst(trace.t, tspan[1])
    integrator = particle.integrator
    reinit!(integrator, integrator.u, t0=tspan[1], tf=tspan[2], reinit_cache=false)
    advance_ssa_sde!(agg, system.reactions, integrator, system.sde_species_mapping, tspan[2], trace, nothing)
    HybridParticle(agg, integrator)
end

PWS.discrete_times(system::MarkovJumpSystem) = range(system.tspan[1], system.tspan[2], step=system.dt)
PWS.discrete_times(system::HybridJumpSystem) = range(system.tspan[1], system.tspan[2], step=system.dt)
PWS.discrete_times(setup::SMC.Setup{<:Trace,<:MarkovJumpSystem}) = PWS.discrete_times(setup.ensemble)
PWS.discrete_times(setup::SMC.Setup{<:Trace,<:HybridJumpSystem}) = PWS.discrete_times(setup.ensemble)


function PWS.generate_configuration(system::HybridJumpSystem; rng=Random.default_rng())
    dtimes = PWS.discrete_times(system)
    traj = zeros(eltype(system.u0), (length(system.u0), length(dtimes)))
    agg, trace = generate_trace(system; traj=traj, rng=rng)
    TraceAndTrajectory(trace, Vector(dtimes), traj, PWS.SSA.speciesnames(system.reactions))
end

function PWS.marginal_density(system::HybridJumpSystem, algorithm, conf::TraceAndTrajectory; full_result=false, kwargs...)
    traced_reactions = system.output_reactions
    trace = filter_trace(ReactionTrace(conf.trace), traced_reactions)
    marginalization_result = PWS.simulate(algorithm, trace, system; Particle=HybridParticle, kwargs...)

    if full_result
        marginalization_result
    else
        PWS.log_marginal(marginalization_result)
    end
end

function PWS.conditional_density(system::HybridJumpSystem, algorithm, conf::TraceAndTrajectory; full_result=false, kwargs...)
    traced_reactions = system.output_reactions
    trace = filter_trace(conf.trace, traced_reactions)
    if 1:num_reactions(system.reactions) ⊆ traced_reactions
        # we don't need to marginalize
        log_probability(system, trace; kwargs...)
    else
        marginalization_result = PWS.simulate(algorithm, trace, system; Particle=MarkovParticle, kwargs...)
        if full_result
            marginalization_result
        else
            PWS.log_marginal(marginalization_result)
        end
    end
end

end # module