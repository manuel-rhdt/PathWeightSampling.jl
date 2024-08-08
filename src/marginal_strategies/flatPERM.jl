module FlatPerm

export PERM

import ..PathWeightSampling: AbstractSimulationAlgorithm, SimulationResult, simulate, discrete_times, logmeanexp, log_marginal, name
import ..SMC: Setup, propagate!, weight, clone, AbstractParticle, spawn

using Statistics
using Random

import LogExpFunctions: logaddexp, logsumexp

# ---------------------------------------------------------------------------- #
#                                CURSOR CONTROL                                #
# ---------------------------------------------------------------------------- #
"""
Save the current cursor position
"""
savecursor(io::IO=stdout) = write(io, "\e[s")

"""
Restore a previously saved cursor position
"""
restorecursor(io::IO=stdout) = write(io, "\e[u")

"""
    clear(io::IO = stdout)
Clear terminal from anything printed in the REPL.
"""
clear(io::IO=stdout) = write(io, "\e[2J")
cleartoend(io::IO=stdout) = write(io, "\e[0J")

"""
PERM(num_chains::Int)

Compute the marginal trajectory probability using flatPERM.
"""
struct PERM <: AbstractSimulationAlgorithm
    num_chains::Int
end

name(x::PERM) = "PERM"

const MAX_COPIES = 25

function grow(
    p::AbstractParticle,
    n::Int,
    n_ind::Int,
    n_chain::Int,
    log_w::Float64,
    log_marginal_estimate::Matrix{Float64},
    num_samples::Matrix{Int},
    num_samples_eff::Matrix{Float64},
    setup;
    inspect=identity
)
    dtimes = discrete_times(setup.ensemble)
    if (n + 1) > length(dtimes)
        return
    end
    tspan = (dtimes[n], dtimes[n+1])

    propagate!(p, tspan, setup)
    inspect(p)
    n += 1 # we increased the size through propagation
    n_ind += 1 # we made an "independent" step
    log_w = log_w + weight(p) # we increased the weight through propagation

    # m = methylation_level(setup.ensemble.reactions, p.agg.u) + 1
    m = 1

    # restorecursor()
    # savecursor()
    # cleartoend()
    # print(UnicodePlots.heatmap(num_samples))

    num_samples[n, m] += 1
    eff_n = num_samples_eff[n, m] += n_ind / n
    cur_estimate = log_marginal_estimate[n, m]
    new_estimate = logaddexp(cur_estimate, log_w - log(n_chain))
    log_marginal_estimate[n, m] = new_estimate

    log_ratio = log_w - new_estimate + log(n_chain) - log(eff_n)
    if log_ratio >= 0
        # enrich
        num_copies = min(trunc(Int, exp(log_ratio)), MAX_COPIES)
        adjusted_weight = log_w - log(num_copies)
        for i in 2:num_copies
            child = clone(p, setup)
            grow(
                child,
                n,
                0,
                n_chain,
                adjusted_weight,
                log_marginal_estimate,
                num_samples,
                num_samples_eff,
                setup;
                inspect
            )
        end
        grow(
            p,
            n,
            0,
            n_chain,
            adjusted_weight,
            log_marginal_estimate,
            num_samples,
            num_samples_eff,
            setup;
            inspect
        )
    else
        # maybe prune
        v = -randexp()
        if v <= log_ratio
            # this happens with probability r=exp(log_ratio)
            # prune failed, continue growing
            grow(
                p,
                n,
                n_ind,
                n_chain,
                new_estimate,
                log_marginal_estimate,
                num_samples,
                num_samples_eff,
                setup;
                inspect
            )
        else
            # this happens with probability (1-r)
            # prune
            return
        end
    end
end

struct PERMResult <: SimulationResult
    log_marginal_estimate::Matrix{Float64}
    num_samples::Matrix{Int}
    num_samples_eff::Matrix{Float64}
end

log_marginal(result::PERMResult) = vcat(0.0, logsumexp(result.log_marginal_estimate, dims=2)[2:end, 1])

# This is the main routine to compute the marginal probability in flatPERM.
function flatperm(convergence_criterion, setup; Particle, inspect=identity)

    # this determines the time-interfaces at which we enrich or prune
    dtimes = discrete_times(setup.ensemble)

    # M = max_methylation_level(setup.ensemble.reactions, setup.ensemble.u0) + 1
    M = 1

    log_marginal_estimate = zeros(length(dtimes), M)
    num_samples = zeros(Int, length(dtimes), M)
    num_samples_eff = zeros(length(dtimes), M)

    # savecursor()
    N = 1
    while mean(num_samples_eff[2:end, :]) < convergence_criterion
        log_marginal_estimate .+= log((N - 1) / N)
        n = 1
        p = spawn(Particle, setup)
        grow(p, n, 0, N, 0.0, log_marginal_estimate, num_samples, num_samples_eff, setup; inspect)
        N += 1
    end

    PERMResult(log_marginal_estimate, num_samples, num_samples_eff)
end

function simulate(algorithm::PERM, initial, system; rng=Random.default_rng(), kwargs...)
    setup = Setup(initial, system, rng)
    flatperm(algorithm.num_chains, setup; kwargs...)
end

end # module