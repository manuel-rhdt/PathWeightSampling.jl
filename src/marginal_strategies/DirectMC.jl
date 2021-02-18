struct DirectMCEstimate
    num_samples::Int
end

name(x::DirectMCEstimate) = "Direct MC"

struct DirectMCResult <: SimulationResult
    samples::Matrix{Float64}
end

log_marginal(result::DirectMCResult) = vec(logmeanexp(result.samples, dims=2))

function simulate(algorithm::DirectMCEstimate, initial, system)
    samples = zeros(Float64, algorithm.num_samples)
    for i in 1:algorithm.num_samples
        signal = new_signal(initial, system)
        samples[i] = -energy(signal, system, 1.0)
    end
    DirectMCResult(samples)
end