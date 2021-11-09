include("basic_setup.jl")

using Statistics

function cost(num_samples, integration_nodes, chain_length)
    algorithm = TIEstimate(1024, integration_nodes, chain_length)
    result = Trajectories.marginal_entropy(gen, algorithm=algorithm; num_samples=num_samples, duration=100.0)
    estimator_var = var(result["marginal_entropy"].Sample) / (num_samples - 1)
    time = sum(result["marginal_entropy"].TimeElapsed)
    estimator_var * time
end

using Hyperopt

ho = @hyperopt for i in 50,
            sampler in GPSampler(Min),
            integration_nodes in 4.0:8.0,
            chain_length in 2.0.^(9:0.125:13)
    @show integration_nodes chain_length
    @show cost(200, round(Int, integration_nodes), round(Int, chain_length))
end

using Plots
p = plot(ho, dpi=300)
savefig(p, projectdir("plots", "hyperopt.png"))

using JSON

filename = projectdir("data", "hyperopt.json")

open(filename, "w") do io
    JSON.print(io, Dict(
    "iterations" => ho.iterations,
    "params" => ho.params,
    "candidates" => ho.candidates,
    "history" => ho.history,
    "results" => ho.results
    ))
end
