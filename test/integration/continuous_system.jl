"""
    ContinuousSystem.jl

Tests for hybrid continuous-discrete stochastic systems (HybridContinuousSystem).
Validates systems with pure continuous SDE evolution and discrete jump events.
Tests state trajectories, equilibrium properties, and mutual information computation.
"""

import PathWeightSampling as PWS
import PathWeightSampling.SSA: AbstractJumpRateAggregator
import PathWeightSampling.ContinuousSystem
using Test
using StaticArrays
using Statistics
using StochasticDiffEq
using Accessors
using DataFrames

import Random

function det_evolution_x(u, p, t)
    κ, λ, ρ, μ = p
    SA[0.0, ρ*u[1] - μ*u[2]]
end

function noise_x(u, p, t)
    κ, λ, ρ, μ = p
    SA[0.0, sqrt(2ρ * 50.0)]
end

κ = 50.0
λ = 1.0
ρ = 10.0
μ = 10.0
ps = (κ, λ, ρ, μ)
u0 = round.(SA[κ / λ, κ * ρ / λ / μ])
tspan = (0.0, 1000.0)
s_prob = SDEProblem(det_evolution_x, noise_x, u0, tspan, ps)

rates = [κ, λ]
rstoich = [Pair{Int, Int}[], [1=>1]]
nstoich = [[1=>1], [1=>-1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])

system = PWS.HybridContinuousSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    tspan,
    0.01, # dt
    s_prob,
    0.01, # sde_dt
    :S,
    :X,
    [1 => 1]
)

conf = PWS.generate_configuration(system)

@test PWS.discrete_times(system) == conf.t
@test round.(conf[1, :]) ≈ conf[1, :] atol=1e-10

@test mean(conf[1, :]) ≈ κ / λ rtol=0.05
@test var(conf[1, :]) ≈ κ / λ rtol=0.1
@test mean(conf[2, :]) ≈ ρ * κ / λ / μ rtol=0.05
@test var(conf[2, :]) ≈ (1 / (1 + λ/μ) + ρ / μ) * κ / λ rtol=0.1

om_int = ContinuousSystem.OMIntegrator(system.sde_prob.f, system.sde_prob.g, system.sde_prob.p, copy(system.u0), 0.0, 1)
agg = PWS.SSA.initialize_aggregator!(
    copy(system.agg),
    system.reactions,
    u0 = copy(system.u0),
    tspan = extrema(PWS.discrete_times(system))
)
@test agg.u == system.u0
@test om_int.u == system.u0
@test agg.tprev == 0.0
ContinuousSystem.step_ssa!(om_int, agg, conf, system)
@test om_int.sol_index == 2
@test om_int.u == conf.u[2]

ContinuousSystem.step_ssa!(om_int, agg, conf, system)
@test om_int.sol_index == 3
@test om_int.u == conf.u[3]

cond_p = PWS.conditional_density(system, PWS.SMCEstimate(128), conf)
marg_p = PWS.marginal_density(system, PWS.SMCEstimate(128), conf)
@test length(marg_p) == length(cond_p) == length(PWS.discrete_times(system))
@test cond_p[end] > marg_p[end]


system = PWS.HybridContinuousSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    (0.0, 10.0),
    0.02, # dt
    s_prob,
    0.01, # sde_dt
    :S,
    :X,
    [1 => 1]
)

result_object = PWS.mutual_information(system, PWS.SMCEstimate(128), num_samples=1000)

pws_result = combine(
    groupby(result_object.result, :time), 
    :MutualInformation => mean => :MI,
    :MutualInformation => sem => :Err
)

relative_error = pws_result.Err[end] / pws_result.MI[end]
@test relative_error < 5e-2

rate_estimate = (pws_result.MI[end] - pws_result.MI[10]) / (pws_result.time[end] - pws_result.time[10])
rate_analytical = λ / 2 * (sqrt(1 + ρ / λ) - 1) # From Tostevin

@test rate_estimate ≈ rate_analytical rtol=5*relative_error