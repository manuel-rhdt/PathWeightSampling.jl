include("basic_setup.jl")
using Statistics

function estimator_variance(num_samples, integration_nodes, chain_length)
    algorithm = TIEstimate(1024, integration_nodes, chain_length)
    result = Trajectories.marginal_entropy(gen, algorithm=algorithm; num_samples=num_samples, duration=100.0)
    estimator_var = var(result["marginal_entropy"].Sample) / (num_samples - 1)
    time = sum(result["marginal_entropy"].TimeElapsed)
    (estimator_var, time)
end

integration_nodes = 4

chains = map(x -> round(Int, x), 2 .^ (8:0.25:11))

results = estimator_variance.(100, integration_nodes, chains)
times = map(x->x[2], results)
variance = map(x->x[1], results)

using DataFrames, GLM
data = DataFrame(N=chains, Time=times, Variance=variance)

ols = lm(@formula(Time ~ N), data)
print(ols)

fixtime = coef(ols)[1] / 100
tau_s = coef(ols)[2] / 100

nr(ns) = round(Int, 10 * 60 / (fixtime + ns * tau_s))

chains2 = map(x -> round(Int, x), 2 .^ (9:0.125:12))
nr.(chains2)

results2 = estimator_variance.(nr.(chains2), integration_nodes, chains2)

using Plots
plot(data.N, [data.Time, predict(ols)])

data2 = DataFrame(N=chains2, Time=map(x->x[2], results2), Variance=map(x->x[1], results2))



plot(data2.N, data2.Variance, xlim=(0, :auto), ylim=(0, :auto))

using DrWatson
using CSV

mkpath(projectdir("data", "estimator_variance"))
CSV.write(projectdir("data", "estimator_variance", "integration_nodes=4.csv"), data2)
CSV.write(projectdir("data", "estimator_variance", "timing_integration_nodes=4.csv"), data)
