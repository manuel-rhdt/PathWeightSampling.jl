
module ContinuousSystem

export HybridContinuousSystem

import ..PathWeightSampling as PWS
import ..PathWeightSampling: AbstractSystem

using ..SSA
using ..SMC

import Random

using StochasticDiffEq
using Accessors

function SSACallback(agg::AbstractJumpRateAggregator, reactions::SSA.AbstractJumpSet, ssa_species_mapping)
    ref_agg = Ref(agg)

    condition = function (u, t, integrator)
        t == ref_agg[].tstop
    end

    affect! = function (integrator)
        agg = ref_agg[]
        tnow = integrator.t
        rx = SSA.select_reaction(agg)
        agg = SSA.perform_jump(agg, rx, reactions)
        for (ssa_index, sde_index) in ssa_species_mapping
            u = integrator.u
            u = @set u[sde_index] = agg.u[ssa_index]
            integrator.u = u
        end
        agg = SSA.update_rates(agg, reactions, rx)
        @reset agg.tstop = tnow + Random.randexp(agg.rng) / agg.sumrate
        if agg.tstop < agg.tspan[2]
            add_tstop!(integrator, agg.tstop)
        end
        ref_agg[] = agg
    end

    initialize_cb = function (c, u, t, integrator)
        agg = SSA.initialize_aggregator(ref_agg[], reactions, u0=u, tspan=integrator.sol.prob.tspan)
        if agg.tstop < agg.tspan[2]
            add_tstop!(integrator, agg.tstop)
        end
        ref_agg[] = agg
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
    HybridContinuousSystem(agg, reactions, u0, tspan, dt, sde_prob, sde_dt, input_reactions, callback)
end

PWS.discrete_times(system::HybridContinuousSystem) = range(system.tspan[1], system.tspan[2], step=system.dt)


function PWS.generate_configuration(system::HybridContinuousSystem; rng=Random.default_rng())
    dtimes = PWS.discrete_times(system)
    seed = rand(rng, UInt)
    solve(system.sde_prob, EM(), dt=system.sde_dt, save_everystep=true, saveat=dtimes, callback = system.callback, seed=seed)
end

struct SDEParticle{Agg,Integrator} <: SMC.AbstractParticle
    agg::Agg
    integrator::Integrator
end

function SDEParticle(setup::SMC.Setup)
    system = setup.ensemble

    tspan = system.tspan
    u0 = system.u0

    agg = initialize_aggregator(
        copy(system.agg),
        system.reactions,
        u0=copy(u0),
        tspan=tspan,
        traced_reactions=BitSet(),
        seed=rand(setup.rng, UInt)
    )

    s_prob = remake(system.sde_prob)
    sde_dt = system.sde_dt
    seed = rand(setup.rng, UInt)
    integrator = init(s_prob, EM(), dt=sde_dt, save_everystep=false, save_start=false, save_end=false, seed=seed)

    HybridParticle(agg, integrator)
end

function SMC.propagate!(particle::SDEParticle, tspan::Tuple{T,T}, setup::SMC.Setup) where {T<:Real}
    system = setup.ensemble
    trace = setup.configuration
    agg = particle.agg
    @reset agg.tprev = tspan[1]
    integrator = particle.integrator
    reinit!(integrator, integrator.u, t0=tspan[1], tf=tspan[2], reinit_cache=false)
    agg = advance_ssa_sde(agg, system.reactions, integrator, system.sde_species_mapping, tspan[2], trace, nothing)
    HybridParticle(agg, integrator)
end

end