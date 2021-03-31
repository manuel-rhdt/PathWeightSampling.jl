import GaussianMcmc
using GaussianMcmc: SMCEstimate, propagate, DirectMCEstimate, marginal_configuration, ConditionalEnsemble, MarginalEnsemble, energy_difference, generate_configuration, log_marginal, logpdf, simulate
using Test

system = GaussianMcmc.chemotaxis_system(dtimes=0:0.05:1.0)
conf = generate_configuration(system)
mconf = marginal_configuration(conf)
cens = ConditionalEnsemble(system)
mens = MarginalEnsemble(system)

@test energy_difference(conf, cens) == energy_difference(mconf, mens)

system.u0
conf.x_traj

@run propagate(conf, cens, cens.jump_problem.prob.u0, (0.0,1.0))
