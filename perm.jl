import PathWeightSampling as PWS

import Random
using StaticArrays

using Plots

import LogExpFunctions: logaddexp, logsumexp

rates = SA[50.0, 1.0, 1.0, 1.0]
rstoich = SA[Pair{Int, Int}[], [1 => 1], [1 => 1], [2 => 1]]
nstoich = SA[[1 => 1], [1 => -1], [2 => 1], [2 => -1]]
reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])
u0 = SA[50.0, 50.0]
tspan = (0.0, 10.0)
system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    u0,
    tspan,
    :S,
    :X
)

conf = PWS.generate_configuration(system; rng=Random.Xoshiro(1))


conditional = PWS.conditional_density(system, PWS.DirectMCEstimate(1), conf)

trace = conf.trace
traced_reactions = system.output_reactions
trace = PWS.filter_trace(trace, traced_reactions)


algorithm = PWS.PERM(1000, 0.1, conditional)
marginalization_result = PWS.simulate(algorithm, trace, system; Particle=PWS.JumpSystem.MarkovParticle)

heatmap(marginalization_result.logZ)
heatmap(log.(marginalization_result.num_samples))

begin
log_marginal = PWS.log_marginal(marginalization_result)
p = plot(0 .* log_marginal)
for i=1:10
    log_marginal2 = PWS.marginal_density(system, PWS.SMCEstimate(512), conf)
    plot!(p, log_marginal2 - log_marginal)
end
p
end