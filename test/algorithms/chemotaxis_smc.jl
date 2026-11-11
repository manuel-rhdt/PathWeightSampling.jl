"""
    chemotaxis_smc.jl

Tests for the aggregated particle filter for the chemotaxis system
(`AggregatedSMCEstimate`, `information_bounds`). The estimates must agree with the generic
`SMCEstimate`, which simulates every reaction, and with their own conditional-SMC bounds.
"""

import PathWeightSampling as PWS
using Test
using Statistics

import Random: Xoshiro

system = PWS.chemotaxis_system(n=3, n_clusters=10, duration=2.0, dt=0.1)
conf = PWS.generate_configuration(system, rng=Xoshiro(1))
ntimes = length(PWS.discrete_times(system))

@testset "interface" begin
    alg = PWS.AggregatedSMCEstimate(512)
    @test PWS.name(alg) == "AggSMC-512"
    cond = PWS.conditional_density(system, alg, conf; rng=Xoshiro(2))
    marg = PWS.marginal_density(system, alg, conf; rng=Xoshiro(3))
    @test length(cond) == ntimes && length(marg) == ntimes
    @test cond[1] == 0.0 && marg[1] == 0.0
    full = PWS.conditional_density(system, alg, conf; rng=Xoshiro(2), full_result=true)
    @test PWS.log_marginal(full) == cond
    @test size(PWS.to_dataframe(full), 1) == ntimes
    # mutual_information end to end
    mi = PWS.mutual_information(system, alg; num_samples=2, progress=false)
    @test size(mi.result, 1) == 2 * ntimes
end

@testset "agreement with generic SMCEstimate" begin
    # The generic filter simulates every reaction; it is a stochastic lower bound of ln P and
    # is biased low by O(1) nats for few particles, hence the additive tolerance.
    short = PWS.chemotaxis_system(n=3, n_clusters=10, duration=1.0, dt=0.1)
    sconf = PWS.generate_configuration(short, rng=Xoshiro(1))
    nrep = 8
    agg = PWS.AggregatedSMCEstimate(4096)
    gen = PWS.SMCEstimate(512)
    runs(alg, density) = [density(copy(short), alg, sconf; rng=Xoshiro(100 + i))[end] for i in 1:nrep]
    for density in (PWS.conditional_density, PWS.marginal_density)
        a, g = runs(agg, density), runs(gen, density)
        se = sqrt(var(a) / nrep + var(g) / nrep)
        @test abs(mean(a) - mean(g)) < 4 * se + 1.5
        @test std(a) < std(g) + 0.5  # the aggregated filter is at least as precise
    end
end

@testset "bounds" begin
    for seed in 1:3
        c = PWS.generate_configuration(system, rng=Xoshiro(seed))
        b = PWS.ChemotaxisSMC.information_bounds(system, c, 8192; rng=Xoshiro(10 + seed))
        @test isfinite(b.lower) && isfinite(b.upper)
        # converged: bounds and estimate agree within Monte Carlo noise (nats)
        @test b.upper - b.lower < 2.0
        @test b.lower - 1.0 < b.smc < b.upper + 1.0
    end
end
