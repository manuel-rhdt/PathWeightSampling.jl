struct DirectMCEstimate
    num_samples::Int
end

name(x::DirectMCEstimate) = "Direct MC"

struct DirectMCResult{Samples} <: SimulationResult
    samples::Samples
end

log_marginal(result::DirectMCResult{<:AbstractMatrix}) = vec(logmeanexp(result.samples, dims=2))
log_marginal(result::DirectMCResult{<:AbstractVector}) = logmeanexp(result.samples)


function simulate(algorithm::DirectMCEstimate, initial, system; kwargs...)
    samples = zeros(Float64, algorithm.num_samples)
    for i in 1:algorithm.num_samples
        signal = sample(initial, system)
        samples[i] = -energy_difference(signal, system)
    end
    DirectMCResult(samples)
end