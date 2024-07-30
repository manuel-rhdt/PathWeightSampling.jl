import PathWeightSampling as PWS
using Test
using Accessors
using StaticArrays

import Random
import Random: Xoshiro

κ = 50.0
λ = 1.0

species = [:X, :Y]
rates = SA[κ, λ]
rstoich = (SA{Pair{Int, Int}}[], SA[1 => 1])
nstoich = (SA[1 => 1], SA[1 => -1])

bd_reactions = PWS.ReactionSet(rates, rstoich, nstoich, species)

u0 = SA[50.0, 0.0]
tspan = (0.0, 3.0)

bd_system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    bd_reactions,
    u0,
    tspan,
    :Y,
    :X,
    0.1
)

trace = PWS.ReactionTrace([1.0, 2.0], [1, 2], BitSet([1, 2]))

p_wait(s, dt) = exp(-(κ + s * λ) * dt)

wait1 = log(p_wait(50, 1.0))
wait2 = log(p_wait(51, 1.0))
wait3 = log(p_wait(50, 1.0))
reac1 = log(κ)
reac2 = log(51 * λ)

function compute_logpdf(system, trace, tspan, u0=u0)
    agg = PWS.initialize_aggregator(system.agg, system.reactions, u0=copy(u0), tspan=tspan, rng=Xoshiro(1), active_reactions=BitSet())
    agg = @set agg.trace_index = searchsortedfirst(trace.t, tspan[1])
    agg = PWS.JumpSystem.advance_ssa(agg, system.reactions, tspan[2], trace, nothing)
    agg.weight
end

function compute_logpdf2(system, trace, tspan)
    setup = PWS.SMC.Setup(trace, system, Random.Xoshiro(1))
    particle = PWS.JumpSystem.MarkovParticle(setup)
    particle = PWS.SMC.propagate(particle, tspan, setup)
    PWS.SMC.weight(particle)
end

for t1 in 0:0.5:3, t2 in t1:0.5:3
    @test compute_logpdf(bd_system, trace, (t1, t2)) == compute_logpdf2(bd_system, trace, (t1, t2))
end

@test compute_logpdf(bd_system, trace, (0.0, 3.0)) ≈ sum((wait1, wait2, wait3, reac1, reac2))
@test compute_logpdf(bd_system, trace, (1.0, 2.0)) ≈ sum((reac1, wait2))

@test PWS.JumpSystem.log_probability(bd_system, trace) ≈ [
    compute_logpdf(bd_system, trace, (0.0, τ)) for τ in PWS.discrete_times(bd_system)
]

cumulative = [compute_logpdf(bd_system, trace, (0.0, τ)) for τ in 0:0.5:3]
@test cumulative ≈ [
    0.0,
    0.5wait1,
    wait1,
    wait1 + reac1 + 0.5wait2,
    wait1 + reac1 + wait2,
    wait1 + reac1 + wait2 + reac2 + 0.5wait3,
    wait1 + reac1 + wait2 + reac2 + wait3,
]

@test cumulative ≈ PWS.JumpSystem.log_probability(bd_system, trace, dtimes = 0:0.5:3)

cumulative = [compute_logpdf(bd_system, trace, (2.0, τ),  SA[51.0, 0.0]) for τ in 2:0.33:3]
@test cumulative ≈ [
    0.0,
    reac2 + 0.33wait3,
    reac2 + 0.66wait3,
    reac2 + 0.99wait3
]

@test cumulative ≈ PWS.JumpSystem.log_probability(bd_system, trace, u0=SA[51.0, 0.0], dtimes = 2:0.33:3)

dtimes = [1.0, 2.0]
u = [[51.0, 0.0], [50.0, 0.0]]
hybrid_trace = PWS.SSA.HybridTrace(Float64[], Int16[], BitSet([1, 2]), u, dtimes, [1 => 1])

@test PWS.JumpSystem.log_probability(bd_system, hybrid_trace, dtimes=0:0.5:3) == [
    0.0,
    0.5wait1,
    wait1,
    wait1 + 0.5wait2,
    wait1 + wait2,
    wait1 + wait2 + 0.5wait3,
    wait1 + wait2 + wait3,
]

@test PWS.JumpSystem.log_probability(bd_system, hybrid_trace, dtimes=2:0.5:3) == [
    0.0,
    0.5wait3,
    wait3
]