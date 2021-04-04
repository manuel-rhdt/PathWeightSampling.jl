import GaussianMcmc
using GaussianMcmc: SMCEstimate, propagate, DirectMCEstimate, marginal_configuration, ConditionalEnsemble, MarginalEnsemble, energy_difference, generate_configuration, log_marginal, logpdf, simulate
using Test

system = GaussianMcmc.chemotaxis_system(dtimes=0:0.05:1.0)
conf = generate_configuration(system)
mconf = marginal_configuration(conf)
cens = ConditionalEnsemble(system)
mens = MarginalEnsemble(system)

@test energy_difference(conf, cens) == energy_difference(mconf, mens)

alg = SMCEstimate(16)
cresult = simulate(alg, conf, cens)
mresult = simulate(alg, mconf, mens)
