module DirectMC

export DirectMCEstimate

import ..PathWeightSampling: AbstractSimulationAlgorithm, SimulationResult, simulate
import ..SMC: SMCEstimate

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
struct DirectMCEstimate <: AbstractSimulationAlgorithm
    num_samples::Int
end

name(x::DirectMCEstimate) = "Direct MC"

function simulate(algorithm::DirectMCEstimate, initial, system; kwargs...)
    estimate = SMCEstimate(algorithm.num_samples)
    simulate(estimate, initial, system; resample_threshold=0, kwargs...)
end

end # module