
module ContinuousSystem

export HybridContinuousSystem

import ..PathWeightSampling as PWS
import ..PathWeightSampling: AbstractSystem
import ..PathWeightSampling.JumpSystem

using ..SSA
using ..SMC

import Random

using StaticArrays
using StochasticDiffEq
using Accessors
using DataFrames

@inline function update_integrator_from_ssa!(integrator, ssa_u, ssa_species_mapping)
    if integrator.u isa StaticArray
        u = integrator.u
        for (ssa_index, sde_index) in ssa_species_mapping
            u = @set u[sde_index] = ssa_u[ssa_index]
        end
        integrator.u = u
    else
        for (ssa_index, sde_index) in ssa_species_mapping
            integrator.u[sde_index] = ssa_u[ssa_index]
        end
    end
end

function SSACallback(agg::AbstractJumpRateAggregator, reactions::SSA.AbstractJumpSet, ssa_species_mapping)

    condition = function (u, t, integrator)
        t == agg.tstop
    end

    affect! = function (integrator)
        tnow = integrator.t
        rx = SSA.select_reaction(agg)
        SSA.perform_jump!(reactions, agg, rx)
        update_integrator_from_ssa!(integrator, agg.u, ssa_species_mapping)
        SSA.update_rates!(agg, reactions, rx)
        agg.tstop = tnow + Random.randexp(agg.rng) / agg.sumrate
        if agg.tstop < agg.tspan[2]
            add_tstop!(integrator, agg.tstop)
        end
    end

    initialize_cb = function (c, u, t, integrator)
        SSA.initialize_aggregator!(agg, reactions, u0=u, tspan=integrator.sol.prob.tspan)
        if agg.tstop < agg.tspan[2]
            add_tstop!(integrator, agg.tstop)
        end
    end

    DiscreteCallback(condition, affect!, initialize=initialize_cb, save_positions=(false, false))
end

struct HybridContinuousSystem{A,JS,U,Prob} <: AbstractSystem
    agg::A
    reactions::JS
    u0::U
    tspan::Tuple{Float64,Float64}
    dt::Float64
    sde_prob::Prob
    sde_dt::Float64
    input_reactions::BitSet
    callback::DiscreteCallback
    ssa_species_mapping::Vector{Pair{Int, Int}}
end

function HybridContinuousSystem(
    alg::AbstractJumpRateAggregatorAlgorithm,
    reactions::AbstractJumpSet,
    u0::AbstractVector,
    tspan::Tuple{Float64,Float64},
    dt::Real,
    sde_prob::SciMLBase.SDEProblem,
    sde_dt::Real,
    input_species::Symbol,
    output_species::Symbol,
    ssa_species_mapping::Vector{Pair{Int,Int}}
)
    ridtogroup = make_reaction_groups(reactions)
    input_reactions = reactions_that_mutate_species(reactions, input_species)
    output_reactions = reactions_that_mutate_species(reactions, output_species)
    @assert output_reactions == Set() "we do not support reactions that mutate the output"
    agg = build_aggregator(alg, reactions, u0, ridtogroup)
    callback = SSACallback(agg, reactions, ssa_species_mapping)
    HybridContinuousSystem(
        agg, 
        reactions, 
        u0, 
        tspan, 
        dt, 
        sde_prob, 
        sde_dt, 
        input_reactions, 
        callback, 
        ssa_species_mapping
    )
end

PWS.discrete_times(system::HybridContinuousSystem) = range(system.tspan[1], system.tspan[2], step=system.dt)


function PWS.generate_configuration(system::HybridContinuousSystem; rng=Random.default_rng())
    dtimes = PWS.discrete_times(system)
    seed = rand(rng, UInt)
    solve(system.sde_prob, EM(), dt=system.sde_dt, save_everystep=false, saveat=dtimes, callback = system.callback, seed=seed)
end

function PWS.to_dataframe(conf::StochasticDiffEq.RODESolution)
    cols = Any[:time=>conf.t]
    for species in axes(conf, 1)
        push!(cols, Symbol(species) => conf[species, :])
    end
    DataFrame(cols)
end

mutable struct OMIntegrator{F, G, P, uType, tType <: Real}
    const f::F
    const g::G
    const p::P

    u::uType
    t::tType
    sol_index::Int
end

function step!(integrator::OMIntegrator, conf::StochasticDiffEq.RODESolution)
    i = integrator.sol_index + 1
    if i > lastindex(conf.t)
        return nothing
    end

    t = integrator.t
    t_next = conf.t[i]

    b = integrator.f(integrator.u, integrator.p, t)
    σ = integrator.g(integrator.u, integrator.p, t)

    acc = 0.0
    Δt = t_next - t
    for j in eachindex(σ)
        !(σ[j] > 0) && continue
        v = (conf[j, i] - integrator.u[j]) / Δt
        action = 0.5 * ((v - b[j]) / σ[j])^2
        acc -= action * Δt
    end
    
    integrator.t = conf.t[i]
    integrator.sol_index = i

    if integrator.u isa StaticArray
        integrator.u = copy(conf.u[i])
    else
        integrator.u .= conf.u[i]
    end

    acc
end

function step_ssa!(integrator::OMIntegrator, agg::AbstractJumpRateAggregator, conf::StochasticDiffEq.RODESolution, system::HybridContinuousSystem)
    i = integrator.sol_index + 1
    if i > lastindex(conf.t)
        return nothing
    end

    t = integrator.t
    t_next = conf.t[i]

    JumpSystem.advance_ssa!(agg, system.reactions, t, nothing, nothing)
    update_integrator_from_ssa!(integrator, agg.u, system.ssa_species_mapping)

    b = integrator.f(integrator.u, integrator.p, t)
    σ = integrator.g(integrator.u, integrator.p, t)

    acc = 0.0
    Δt = t_next - t
    for j in eachindex(σ)
        !(σ[j] > 0) && continue
        v = (conf[j, i] - integrator.u[j]) / Δt
        action = 0.5 * ((v - b[j]) / σ[j])^2
        acc -= action * Δt
    end
    
    integrator.t = conf.t[i]
    integrator.sol_index = i


    if integrator.u isa StaticArray
        integrator.u = copy(conf.u[i])
    else
        integrator.u .= conf.u[i]
    end

    acc
end

struct SDEParticle{Agg,Integrator} <: SMC.AbstractParticle
    agg::Agg
    integrator::Integrator
    weight::Float64
end

function SMC.spawn(::Type{<:SDEParticle}, setup::SMC.Setup)
    system = setup.ensemble

    tspan = system.tspan
    u0 = system.u0

    agg = initialize_aggregator!(
        copy(system.agg),
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        traced_reactions=BitSet(),
        seed=rand(setup.rng, UInt)
    )

    integrator = OMIntegrator(
        system.sde_prob.f,
        system.sde_prob.g,
        system.sde_prob.p,
        copy(u0),
        tspan[1],
        firstindex(setup.configuration.t)
    )

    SDEParticle(agg, integrator, 0.0)
end

SMC.weight(particle::SDEParticle) = particle.weight

function SMC.propagate!(particle::SDEParticle, tspan::Tuple{T,T}, setup::SMC.Setup) where {T<:Real}
    system = setup.ensemble
    conf = setup.configuration
    particle.integrator.t = tspan[1]
    particle.integrator.sol_index = searchsortedfirst(conf.t, tspan[1])
    val = 0.0
    while particle.integrator.t < tspan[2]
        dv = step_ssa!(particle.integrator, particle.agg, conf, system)
        if dv === nothing
            break
        end
        val += dv
    end
    SDEParticle(particle.agg, particle.integrator, val)
end

function SMC.clone_from!(child::SDEParticle, parent::SDEParticle, setup::SMC.Setup)
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

    child.integrator.u = copy(parent.integrator.u)

    child
end

function log_probability(system::HybridContinuousSystem, conf::StochasticDiffEq.RODESolution; u0=system.u0, dtimes=PWS.discrete_times(system))
    tspan = extrema(dtimes)
    integrator = OMIntegrator(
        system.sde_prob.f,
        system.sde_prob.g,
        system.sde_prob.p,
        copy(u0),
        tspan[1],
        firstindex(conf.t)
    )
    logp = Vector{Float64}(undef, length(dtimes))

    acc = 0.0
    k = 1
    logp[k] = acc
    for t in dtimes[2:end]
        k += 1
        while integrator.t < t
            x = step!(integrator, conf)
            x === nothing && break
            acc += x
        end
        logp[k] = acc
    end
    logp
end

function PWS.marginal_density(system::HybridContinuousSystem, algorithm, conf::StochasticDiffEq.RODESolution; full_result=false, kwargs...)
    marginalization_result = PWS.simulate(algorithm, conf, system; Particle=SDEParticle, kwargs...)

    if full_result
        marginalization_result
    else
        PWS.log_marginal(marginalization_result)
    end
end

function PWS.conditional_density(system::HybridContinuousSystem, algorithm, conf::StochasticDiffEq.RODESolution; kwargs...)
    log_probability(system, conf)
end

end