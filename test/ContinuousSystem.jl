import PathWeightSampling as PWS
using Test
using StaticArrays
using Statistics
using StochasticDiffEq

import Random

function det_evolution(u::SVector, p, t)
    κ, λ, ρ, μ = p
    SA[0.0, ρ*u[1] - μ*u[2]]
end

function noise(u::SVector, p, t)
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
s_prob = SDEProblem(det_evolution, noise, u0, tspan, ps)

rates = (κ, λ)
rstoich = [Pair{Int, Int}[], [1=>1]]
nstoich = [[1=>1], [1=>-1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])

system = PWS.HybridContinuousSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    tspan,
    0.1, # dt
    s_prob,
    0.01, # sde_dt
    :S,
    :X,
    [1 => 1]
)

conf = PWS.generate_configuration(system)

@test PWS.discrete_times(system) ⊆ conf.t

@test mean(conf[1, :]) ≈ κ / λ rtol=0.05
@test var(conf[1, :]) ≈ κ / λ rtol=0.1
@test mean(conf[2, :]) ≈ ρ * κ / λ / μ rtol=0.05
@test var(conf[2, :]) ≈ (1 / (1 + λ/μ) + ρ / μ) * κ / λ rtol=0.1

function log_probability(system, conf; dtimes=PWS.discrete_times(system))
    sde_prob = system.sde_prob
    f = sde_prob.f
    g = sde_prob.g
    p = sde_prob.p
    logp = Vector{Float64}(undef, length(dtimes))

    k = firstindex(dtimes)
    acc = 0.0
    logp[k] = acc
    k += 1
    for i in eachindex(conf.t)[2:end]
        b = f(conf.u[i-1], p, conf.t[i-1])
        σ = g(conf.u[i-1], p, conf.t[i-1])
        Δt = conf.t[i] - conf.t[i-1]
        for j in eachindex(σ)
            σ[j] <= 0 && continue
            v = (conf[j, i] - conf[j, i-1]) / Δt
            action = 0.5 * ((v - b[j]) / σ[j])^2
            acc -= action * Δt - log(σ[j])
        end 

        if k <= lastindex(dtimes) && dtimes[k] == conf.t[i]
            logp[k] = acc
            k += 1
        end
    end
    logp
end

@time log_probability(system, conf)

