using BenchmarkTools
using GaussianMcmc

system = GaussianMcmc.chemotaxis_system()
algorithm = DirectMCEstimate(10_000)
dtimes = collect(range(0.0, 2.0, length=51)[2:end])
initial = generate_configuration(system)

cond_ensemble = GaussianMcmc.ConditionalEnsemble(system)
marg_ensemble = GaussianMcmc.MarginalEnsemble(system)

@benchmark GaussianMcmc.simulate(algorithm, initial, cond_ensemble, dtimes)
@benchmark GaussianMcmc.simulate(algorithm, GaussianMcmc.marginal_configuration(initial), marg_ensemble, dtimes)
