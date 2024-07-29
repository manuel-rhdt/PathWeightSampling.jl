import PathWeightSampling as PWS
using Test
using StaticArrays
using Statistics
using StochasticDiffEq

function det_evolution(u::SVector, p, t)
    κ, λ = p
    SA[κ-λ*u[1]]
end

function noise(u::SVector, p, t)
    κ, λ = p
    SA[sqrt(2κ / λ)]
end

ps = [50.0, 1.0]
tspan = (0.0, 100.0)
s_prob = SDEProblem(det_evolution, noise, SA[50.0], tspan, ps)
sde_species_mapping = [1 => 1]

rates = [1.0, 1.0]
rstoich = [SA[1=>1], SA[2=>1]]
nstoich = [SA[2=>1], SA[2=>-1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])

u0 = SA[50.0, 50.0]
system = PWS.HybridJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    tspan,
    1.0, # dt
    s_prob,
    0.1, # sde_dt
    :S,
    :X,
    sde_species_mapping
)

@test system.output_reactions == Set([1, 2])

conf = PWS.generate_configuration(system)

@test var(conf.traj[1, :]) ≈ 50.0 rtol=0.3

for (i, t) in enumerate(conf.discrete_times)
    if i == 1
        @test conf.traj[1, i] == u0[1]
        continue
    end
    j = searchsortedfirst(conf.trace.dtimes, t)
    @test conf.traj[1, i] == conf.trace.u[j][1]
end
