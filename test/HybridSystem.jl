import PathWeightSampling as PWS
using Test
using StaticArrays
using Statistics
using StochasticDiffEq

import Random

function det_evolution(u::SVector, p, t)
    κ, λ = p
    SA[κ-λ*u[1]]
end

function noise(u::SVector, p, t)
    κ, λ = p
    SA[sqrt(2κ)]
end

κ = 50.0
λ = 1.0
ρ = 10.0
μ = 10.0
ps = [κ, λ]
tspan = (0.0, 100.0)
s_prob = SDEProblem(det_evolution, noise, SA[50.0], tspan, ps)
sde_species_mapping = [1 => 1]

rates = [ρ, μ]
rstoich = [[1=>1], [2=>1]]
nstoich = [[2=>1], [2=>-1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])

u0 = SA[50.0, 50.0]
system = PWS.HybridJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    tspan,
    0.1, # dt
    s_prob,
    0.01, # sde_dt
    :S,
    :X,
    sde_species_mapping
)

@test system.output_reactions == Set([1, 2])

conf = PWS.generate_configuration(system)

@test mean(conf.traj[1, :]) ≈ κ / λ rtol=0.3
@test var(conf.traj[1, :]) ≈ κ / λ rtol=0.3
@test mean(conf.traj[2, :]) ≈ ρ * κ / λ / μ rtol=0.3
@test var(conf.traj[2, :]) ≈ (1 / (1 + λ/μ) + ρ / μ) * κ / λ rtol=0.3

for (i, t) in enumerate(conf.discrete_times)
    if i == 1
        @test conf.traj[1, i] == u0[1]
        continue
    end
    j = searchsortedfirst(conf.trace.dtimes, t)
    @test conf.traj[1, i] == conf.trace.u[j][1]
end

trace = conf.trace
@test trace.traced_reactions == Set([1, 2])

@time "conditional density 1" cd1 = PWS.conditional_density(system, PWS.SMCEstimate(256), conf)
@time "conditional density 2" cd2 = PWS.conditional_density(system, PWS.SMCEstimate(256), conf)
@test cd1 == cd2 == PWS.JumpSystem.log_probability(system, trace) # deterministic evaluation of likelihood

@time "marginal density 1" md1 = PWS.marginal_density(system, PWS.SMCEstimate(256), conf)
@time "marginal density 2" md2 = PWS.marginal_density(system, PWS.SMCEstimate(256), conf)
@test md1 != md2 # MC evaluation of marginal likelihood is non-deterministic
@test all(isapprox.(md1, md2, rtol=1e-3)) # but the results should be very close

@test cd1[end] > md1[end]


system = PWS.HybridJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    (0.0, 10.0),
    0.1, # dt
    s_prob,
    0.01, # sde_dt
    :S,
    :X,
    sde_species_mapping
)

mi = PWS.mutual_information(system, PWS.SMCEstimate(128), num_samples=1000)

using DataFrames, Statistics
sem(x) = sqrt(var(x) / length(x))
pws_result = combine(
    groupby(mi.result, :time), 
    :MutualInformation => mean => :MI,
    :MutualInformation => sem => :Err
)

rate = (pws_result.MI[end] - pws_result.MI[10]) / (pws_result.time[end] - pws_result.time[10])
rate_analytical = λ / 2 * (sqrt(1 + 2ρ / λ) - 1) # From Moor et al. PRR 2023

@test rate ≈ rate_analytical rtol=0.1
