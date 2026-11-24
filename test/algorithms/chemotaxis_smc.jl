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
    @test "ESS" in names(PWS.to_dataframe(full))
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

@testset "multinomial resampler keeps the reference" begin
    rng = Xoshiro(5)
    w = [1.0; fill(1e-6, 8); 5.0]
    for _ in 1:20
        idx = PWS.SMC.multinomial_sample_keep_first(rng, w)
        @test length(idx) == 10 && idx[1] == 1 && issorted(idx) && all(1 .<= idx .<= 10)
    end
    # the heavy particle 10 receives about 5/6 of the draws
    draws = reduce(vcat, [PWS.SMC.multinomial_sample_keep_first(rng, w)[2:end] for _ in 1:2000])
    @test isapprox(count(==(10), draws) / length(draws), 5 / 6, atol=0.02)
end

@testset "conditional SMC" begin
    C = PWS.ChemotaxisSMC
    final(result) = last(PWS.log_marginal(result))
    short = PWS.chemotaxis_system(n=3, n_clusters=10, duration=1.0, dt=0.1)
    sconf = PWS.generate_configuration(short, rng=Xoshiro(1))
    M = C.build_model(short, sconf, 1.0)
    R = C.true_path(M, short, sconf, Xoshiro(7))
    sig = C.signal_logL(M, sconf)
    @test R.logL == sig && length(R.v) == M.K

    # the filters return the standard SMCResult, including the effective sample size
    result = C.run_smc(M, 64, Xoshiro(1))
    @test result isa PWS.SMC.SMCResult
    @test length(result.effective_sample_sizes) == length(PWS.discrete_times(short))
    @test all(1 .<= result.effective_sample_sizes .<= 64)

    # with a single particle the estimate is the reference's own likelihood
    for conditional in (true, false)
        res, refll = C.run_csmc(M, 1, Xoshiro(1), R; conditional)
        @test final(res) ≈ refll rtol = 1e-12
    end

    # E[ln Ẑ_csmc] ≥ ln Z ≥ E[ln Ẑ_smc]; for few particles the difference is large
    nrep = 64
    se(x) = std(x) / sqrt(length(x))
    for conditional in (true, false)
        smc = [final(C.run_smc(M, 32, Xoshiro(i); signal=conditional ? sig : nothing)) for i in 1:nrep]
        csmc = [final(C.run_csmc(M, 32, Xoshiro(1000 + i), R; conditional)[1]) for i in 1:nrep]
        @test mean(csmc) > mean(smc) - 2 * hypot(se(smc), se(csmc))
    end
end
