
include("basic_setup.jl")

function perf_est(subsample)
    algorithm = AnnealingEstimate(subsample, 50, 100)
    Trajectories.simulate(algorithm, initial, system)
end

results = perf_est.(1:5:26)

using Statistics
using Plots

plot(1:5:26, Trajectories.log_marginal.(results), yerr=sqrt.(var.(results)))
plot(1:5:26, sqrt.(var.(results)))


plot(result.inv_temps[1:end - 1], mean(result.acceptance, dims=2))
plot(result.inv_temps[begin + 1:end - 1], diff(mean(result.weights, dims=2), dims=1))

histogram(result.weights[end, :])

Trajectories.log_marginal(result)
sqrt(var(result))

mean(result.weights[end, :])

var(result.weights[end, :]) / 10_000
