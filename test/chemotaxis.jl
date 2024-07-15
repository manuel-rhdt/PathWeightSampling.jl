import PathWeightSampling as PWS
using Test
using Statistics
import Distributed
using Statistics

import Random: Xoshiro

# ChemotaxisJumps

system = PWS.simple_chemotaxis_system(
    n_clusters=800,
    n=6,
    duration=100.0,
    dt=0.1
)
dtimes = PWS.discrete_times(system)

PWS.SSA.make_depgraph(system.reactions)

for (rid, gid) in enumerate(system.agg.ridtogroup)
    if gid == 1
        @test PWS.reaction_type(system.reactions, rid)[1] == 2
    elseif gid == 2
        @test PWS.reaction_type(system.reactions, rid)[1] == 3
    else
        @test PWS.reaction_type(system.reactions, rid)[1] < 2
    end
end

conf = PWS.generate_configuration(system, rng=Xoshiro(2))

k_A = system.reactions.k_A
k_Z = system.reactions.k_Z
a = system.reactions.k_R / (system.reactions.k_R + system.reactions.k_B)
n_clusters = 800
n_chey = 10_000
ϕ_y = 1 / 6

phosphorylation_rate = a * n_clusters * k_A * (1 - ϕ_y) * n_chey + k_Z * ϕ_y * n_chey
rate_estimate = 1 / mean(diff(conf.trace.t))

@test phosphorylation_rate ≈ rate_estimate rtol = 0.05

@test conf.trace isa PWS.HybridTrace
@test PWS.ReactionTrace(conf.trace) isa PWS.ReactionTrace

if haskey(ENV, "PWS_ALL_TESTS")
    println("RUN ALL TESTS")

    systemfn = () -> PWS.simple_chemotaxis_system(
        n_clusters=800,
        n=6,
        duration=10.0,
        dt=0.1
    )

    Distributed.@everywhere import PathWeightSampling

    alg = PWS.SMCEstimate(256)
    mi = PWS.run_parallel(systemfn, alg, 320)

    for x in mean(mi.MutualInformation)
        println(x)
    end
end