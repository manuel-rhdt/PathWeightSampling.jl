include("basic_setup.jl")

import Logging
Logging.disable_logging(Logging.Info)

function estimator_variance(num_responses, integration_nodes, chain_length)
    algorithm = TIEstimate(1024, integration_nodes, chain_length)
    result = Trajectories.marginal_entropy(gen, algorithm=algorithm; num_responses=num_responses, duration=100.0)
    estimator_var = var(result["marginal_entropy"].Sample) / (num_responses - 1)
    time = sum(result["marginal_entropy"].TimeElapsed)
    (estimator_var, time)
end

integration_nodes = 4

chains = map(x -> round(Int, x), 2 .^ (7:0.25:10))

results = estimator_variance.(100, integration_nodes, chains)
times = map(x->x[2], results)
variance = map(x->x[1], results)

using DataFrames, GLM
data = DataFrame(N=chains, Time=times, Variance=variance)

ols = lm(@formula(Time ~ N), data)

fixtime = coef(ols)[1] / 100
tau_s = coef(ols)[2] / 100

nr(ns) = round(Int, 10 * 60 / (fixtime + ns * tau_s))

chains2 = map(x -> round(Int, x), 2 .^ (9:0.125:12))
nr.(chains2)

results2 = estimator_variance.(nr.(chains2), integration_nodes, chains2)


plot(data.N, [data.Time, predict(ols)])

data2 = DataFrame(N=chains2, Time=map(x->x[2], results2), Variance=map(x->x[1], results2))

plot(data2.N, data2.Time, ylim=(0, 40))
