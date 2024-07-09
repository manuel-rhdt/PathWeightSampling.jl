import PathWeightSampling as PWS
using Test
using Setfield

κ = 50.0
λ = 1.0

species = [:X, :Y]
rates = [κ, λ]
rstoich = [[], [1 => 1]]
nstoich = [[1 => 1], [1 => -1]]

bd_reactions = PWS.ReactionSet(rates, rstoich, nstoich, species)

u0 = [50.0]
tspan = (0.0, 3.0)

bd_system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    bd_reactions,
    u0,
    tspan,
    :Y,
    :X
)

trace = PWS.ReactionTrace([1.0, 2.0], [1, 2], BitSet([1, 2]))

p_wait(s, dt) = exp(-(κ + s * λ) * dt)

wait1 = log(p_wait(50, 1.0))
wait2 = log(p_wait(51, 1.0))
wait3 = log(p_wait(50, 1.0))
reac1 = log(κ)
reac2 = log(51 * λ)

function compute_logpdf(system, tspan, u0=u0)
    agg = PWS.initialize_aggregator(system.agg, system.reactions, u0=copy(u0), tspan=tspan, seed=1, active_reactions=BitSet())
    agg = @set agg.trace_index = searchsortedfirst(trace.t, tspan[1])
    agg = PWS.JumpSystem.advance_ssa(agg, system.reactions, tspan[2], trace, nothing)
    agg.weight
end

@test compute_logpdf(bd_system, (0.0, 3.0)) ≈ sum((wait1, wait2, wait3, reac1, reac2))
@test compute_logpdf(bd_system, (1.0, 2.0)) ≈ sum((reac1, wait2))

cumulative = [compute_logpdf(bd_system, (0.0, τ)) for τ in 0:0.5:3]
@test cumulative ≈ [
    0.0,
    0.5wait1,
    wait1,
    wait1 + reac1 + 0.5wait2,
    wait1 + reac1 + wait2,
    wait1 + reac1 + wait2 + reac2 + 0.5wait3,
    wait1 + reac1 + wait2 + reac2 + wait3,
]

cumulative = [compute_logpdf(bd_system, (2.0, τ), [51]) for τ in 2:0.33:3]
@test cumulative ≈ [
    0.0,
    reac2 + 0.33wait3,
    reac2 + 0.66wait3,
    reac2 + 0.99wait3
]

