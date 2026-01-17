module FlatPerm

export PERM

import ..PathWeightSampling: AbstractSimulationAlgorithm, SimulationResult, simulate, discrete_times, logmeanexp, log_marginal, name
import ..SMC: Setup, propagate!, weight, clone, AbstractParticle, spawn

using Statistics
using Random

import LogExpFunctions: logaddexp, logsumexp

"""
PERM(num_chains::Int)

Compute the marginal trajectory probability using flatPERM.
"""
struct PERM <: AbstractSimulationAlgorithm
    num_chains::Int
    alpha::Float64
    conditional::Vector{Float64}
end

name(x::PERM) = "PERM"


struct PERMResult <: SimulationResult
    logZ::Matrix{Float64}
    num_samples::Matrix{Int}
    num_samples_eff::Matrix{Float64}
end

log_marginal(result::PERMResult) = logsumexp(result.logZ, dims=2)[:, 1]

# This is the main routine to compute the marginal probability in flatPERM.
function flatperm(alg::PERM, setup; Particle, inspect=identity)

    # this determines the time-interfaces at which we enrich or prune
    dtimes = discrete_times(setup.ensemble)
    log_conditional = alg.conditional

    # Make energy bins
    M = 50 # number of bins
    ΔW = 1e-1
    α = alg.alpha
    calc_m(W, n) = clamp(round(Int, expm1(α * W / n) / α / ΔW) + M÷2, 1, M)

    n_min_prune = 5  # skip pruning for first 3 steps

    # ln Ẑ[n, m] estimated partition sum. This will be the log marginal probability ln P(x)
    logZ = fill(-Inf, length(dtimes), M)

    # H[n, m] number of visits
    H = zeros(Int, length(dtimes), M)

    stack = Tuple{Particle, Float64, Int}[]
    for S = 1:alg.num_chains
        n = 2
        p = spawn(Particle, setup)
        logW = 0.0
        push!(stack, (p, logW, n))

        while !isempty(stack)
            p, logW, n = pop!(stack)
            
            while n <= length(dtimes)
                tspan = (dtimes[n-1], dtimes[n])

                propagate!(p, tspan, setup)
                inspect(p)

                # update the weight of the configuration
                # note that weight(p) returns the log-likelihood increment
                logW += weight(p) 
                m = calc_m(logW - log_conditional[n], n)

                # --- Update estimators ---
                logZ[n, m] = logaddexp(logW, logZ[n, m])
                H[n, m] += 1

                if n < n_min_prune
                    n += 1
                    # grow without pruning
                    continue
                end

                # --- Compute ratio ---
                if logZ[n, m] == -Inf
                    r = 1
                else
                    r = exp(logW - logZ[n, m] + log(S))
                end

                n += 1 # we increased the size through propagation

                # --- Enrichment ---
                if r > 1
                    k = floor(r + rand())   # stochastic rounding
                    logW -= log(k) # W <- W/k

                    for i = 2:k
                        push!(stack, (clone(p, setup), logW, n))
                    end
                # --- Pruning ---
                else
                    if rand() > r
                        break                      # prune
                    else
                        logW = log(r)               # rescale weight
                    end
                end
            end

            # Finish tour if chain reached full length
            if n > length(dtimes)
                empty!(stack)
                break
            end
        end
    end

    logZ .-= log(alg.num_chains)
    logZ[1, :] .= -log(M)

    PERMResult(logZ, H, zeros(length(dtimes), M))
end

function simulate(algorithm::PERM, initial, system; rng=Random.default_rng(), kwargs...)
    setup = Setup(initial, system, rng)
    flatperm(algorithm, setup; kwargs...)
end

end # module