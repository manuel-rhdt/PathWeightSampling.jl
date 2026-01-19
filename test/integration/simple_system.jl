"""
    SimpleSystem.jl

Tests for simple stochastic systems and mutual information computation.
Validates the chemotaxis system fixture and mutual information estimation
across different sampling algorithms (SMC, Direct MC, PERM).
"""

import PathWeightSampling as PWS
using Test
using Statistics
using DataFrames

println("Test chemotaxis system")
system = PWS.chemotaxis_system(n=3, n_clusters=800, duration=2.0, dt=0.1)
dtimes = PWS.discrete_times(system)

PWS.SSA.num_reactions(system.reactions)

conf = PWS.generate_configuration(system)
@test all(conf.trace.rx .<= PWS.SSA.num_reactions(system.reactions))
@test all(conf.traj .>= 0)

algorithms = [PWS.DirectMCEstimate(128), PWS.SMCEstimate(128), PWS.PERM(16)]
for algorithm in algorithms
    @time "using algorithm $(PWS.name(algorithm))" mi = PWS.mutual_information(system, algorithm, num_samples=4, threads=true, progress=false)
    @test length(mi.result.MutualInformation) == 4 * length(dtimes)
    @test mi.result.MutualInformation[mi.result.time .== 0] == [0, 0, 0, 0]
    for (i, g) in enumerate(groupby(mi.result, :time))
        @test size(g, 1) == 4
        if i > 1
            @test var(g.MutualInformation) > 0
        end
    end
end