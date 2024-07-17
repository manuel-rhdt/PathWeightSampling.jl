
import PathWeightSampling as PWS

using Test

κ = 10.0
λ = 1.0
ρ = 1.0
μ = 1.0
ρ2 = 10.0
μ2 = 10.0

rstoich = [Pair{Int64,Int64}[], [1 => 1]]
nstoich = [[1 => 1], [1 => -1]]
species = [:S]

rates = (
    u -> κ,
    u -> u[1] * λ,
)

jumps = PWS.SSA.ConstantRateJumps(rates, rstoich, nstoich, species)
reaction_groups = PWS.SSA.make_reaction_groups(jumps, :S)
@test reaction_groups == [1, 2]

agg = PWS.build_aggregator(PWS.GillespieDirect(), jumps, reaction_groups)
agg = PWS.initialize_aggregator(agg, jumps)
@test agg.u == [0]
@test agg.rates == [κ, 0.0]

agg = PWS.step_ssa(agg, jumps, nothing, nothing)
@test agg.u == [1]
@test agg.rates == [κ, λ]

# Full system

rstoich = [
    Pair{Int64,Int64}[],
    [1 => 1],
    [1 => 1],
    [2 => 1],
    [2 => 1],
    [3 => 1]
]
nstoich = [
    [1 => 1],
    [1 => -1],
    [2 => 1],
    [2 => -1],
    [3 => 1],
    [3 => -1]
]
species = [:S, :V, :X]

rates = (
    u -> κ,
    u -> u[1] * λ,
    u -> u[1] * ρ,
    u -> u[2] * μ,
    u -> u[2] * ρ2,
    u -> u[3] * μ2
)

jumps = PWS.SSA.ConstantRateJumps(rates, rstoich, nstoich, species)
u0 = [
    κ / λ,
    κ / λ * ρ / μ,
    κ / λ * ρ / μ * ρ2 / μ2
]
tspan = (0.0, 10.0)

system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    jumps,
    u0,
    tspan,
    :S,
    :X,
    1e-2
)

@test system.agg.ridtogroup == [0, 0, 0, 0, 1, 2]
@test system.input_reactions == Set([1, 2])
@test system.output_reactions == Set([5, 6])

conf = PWS.generate_configuration(system)
@test issorted(conf.trace.t)

cond_d = PWS.conditional_density(system, PWS.SMCEstimate(256), conf)
marg_d = PWS.marginal_density(system, PWS.SMCEstimate(256), conf)
@test cond_d[end] >= marg_d[end]

result = PWS.mutual_information(system, PWS.SMCEstimate(128); num_samples=32)

@test length(unique(result["mutual_information"].MutualInformation)) == 32
