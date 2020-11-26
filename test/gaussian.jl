using GaussianMcmc

system = GaussianSystem(delta_t=0.01)
initial = generate_configuration(system)

algorithm = TIEstimate(1024, 6, 2^14)

result = GaussianMcmc.simulate(algorithm, initial, system, scale=0.08)

sqrt(var(result))
log_marginal(result)

plot(result.inv_temps, mean(result.acceptance, dims=1)') 