struct DirectMCEstimate
    num_samples::Int
end

name(x::DirectMCEstimate) = "Direct MC"

struct DirectMCResult <: SimulationResult
    samples::Matrix{Float64}
end

log_marginal(result::DirectMCResult) = logmeanexp(result.samples, dims=2)

function simulate(algorithm::DirectMCEstimate, initial, system, dtimes)
    samples = collect_samples(initial, system, algorithm.num_samples, dtimes)
    DirectMCResult(samples)
end