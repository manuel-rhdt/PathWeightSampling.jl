"""
    ChemotaxisSMC

Fast particle filter (SMC) for the marginal and conditional probability of the output of
`PWS.chemotaxis_system`, with the output likelihood aggregated analytically.

The generic [`SMCEstimate`](@ref) propagates every reaction of every particle with the SSA,
including ~10^4 phosphorylation / dephosphorylation events per second. Here only the
methylation state of each receptor cluster (and, for the marginal, the input signal
`(v, log L)`) is simulated. The output reactions are *not* simulated: given the latent
state, the phosphorylation events are an inhomogeneous Poisson process with intensity
`k_A Y A(t)` (`A` = total kinase activity, piecewise constant between SDE steps and
methylation jumps), while the dephosphorylation events do not depend on the latent state.
Hence the log-weight of a particle over a time piece only needs the number of
phosphorylation events `n` in it and `∫ Y dt`:

    ln w = n ln A − k_A A ∫Y dt        (+ a latent-independent constant `Econst`)

This is the same weight as the SSA-based particle filter, but it is computed with a single
pass over the data, is independent of the output event rate, and uses exact (not
thinned) sampling of the methylation jumps between SDE steps, so large particle numbers
(10^5–10^6) are practical.

The code is split in three parts:

  * `chemotaxis_smc/model.jl`: the system parameters and the aggregated output data,
  * `chemotaxis_smc/filter.jl`: the particle and the filter (`SMC.sample` is the engine),
  * `chemotaxis_smc/bounds.jl`: [`information_bounds`](@ref), a convergence check from a
    conditional SMC that keeps the generating (true) latent path as reference particle.
"""
module ChemotaxisSMC

export AggregatedSMCEstimate, information_bounds

import ..PathWeightSampling as PWS
import ..SMC
import ..PathWeightSampling: AbstractSimulationAlgorithm, discrete_times, log_marginal, name
using ..JumpSystem: HybridJumpSystem, TraceAndTrajectory
using Random

const ChemotaxisSystem = HybridJumpSystem{<:Any,<:PWS.ChemotaxisJumps}

include("chemotaxis_smc/model.jl")
include("chemotaxis_smc/filter.jl")
include("chemotaxis_smc/bounds.jl")

# ----------------------------------------------------------------------------------------
# Integration with the PWS algorithm interface

"""
    AggregatedSMCEstimate(num_particles::Int)

Particle-filter estimate of the marginal and conditional trajectory probability for
`PWS.chemotaxis_system`, using the aggregated output likelihood (see [`ChemotaxisSMC`](@ref)).
It estimates the same quantities as [`SMCEstimate`](@ref) but is much faster for this
system and therefore allows far larger particle numbers.

```julia
system = PWS.chemotaxis_system(n_clusters=25, duration=20.0)
mi = PWS.mutual_information(system, PWS.AggregatedSMCEstimate(2^16); num_samples=8)
```

The estimates (`full_result=true`) are the same `SMCResult` as for `SMCEstimate`, including
the effective sample size.
"""
struct AggregatedSMCEstimate <: AbstractSimulationAlgorithm
    num_particles::Int
end

name(x::AggregatedSMCEstimate) = "AggSMC-$(x.num_particles)"

function _density(system, algorithm::AggregatedSMCEstimate, conf, rng, full_result; conditional)
    M = build_model(system, conf, system.tspan[2])
    result = run_smc(M, algorithm.num_particles, rng; signal=conditional ? signal_logL(M, conf) : nothing)
    full_result ? result : log_marginal(result)
end

PWS.marginal_density(system::ChemotaxisSystem, algorithm::AggregatedSMCEstimate, conf::TraceAndTrajectory;
    full_result=false, rng=Random.default_rng(), kwargs...) =
    _density(system, algorithm, conf, rng, full_result; conditional=false)

PWS.conditional_density(system::ChemotaxisSystem, algorithm::AggregatedSMCEstimate, conf::TraceAndTrajectory;
    full_result=false, rng=Random.default_rng(), kwargs...) =
    _density(system, algorithm, conf, rng, full_result; conditional=true)

end # module
