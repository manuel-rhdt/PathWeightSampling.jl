using BenchmarkTools
using PWS

system = PWS.chemotaxis_system()
algorithm = DirectMCEstimate(10_000)
dtimes = collect(range(0.0, 2.0, length=51)[2:end])
initial = generate_configuration(system)

cond_ensemble = PWS.ConditionalEnsemble(system)
marg_ensemble = PWS.MarginalEnsemble(system)

@benchmark PWS.simulate(algorithm, initial, cond_ensemble, dtimes)
@benchmark PWS.simulate(algorithm, PWS.marginal_configuration(initial), marg_ensemble, dtimes)
