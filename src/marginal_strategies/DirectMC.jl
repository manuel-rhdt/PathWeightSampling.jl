"""
    DirectMCEstimate(M::Int)

This is the simplest marginalization strategy, estimating the marginal path
probabilities using a direct Monte Carlo simulation.

`M` specifies the number of samples to be used for the Monte Carlo average.
Larger `M` improves the accuracy of the marginalization integral, but increases
the computational cost.

# Mathematical Description

The marginal probability
```math
\\mathrm{P}[\\bm{x}] = \\int\\mathrm{d}\\bm{s} \\mathrm{P}[\\bm{x}|\\bm{s}] \\mathrm{P}[\\bm{s}]
```
can be computed via a Monte Carlo estimate by sampling `M` trajectories from
``\\mathrm{P}[\\bm{s}]`` and taking the average of the likelihoods:
```math
\\mathrm{P}[\\bm{x}] = \\langle \\mathrm{P}[\\bm{x}|\\bm{s}] \\rangle_{\\mathrm{P}[\\bm{s}]}\\,.
```
"""
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