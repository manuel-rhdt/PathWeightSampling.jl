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
end

name(x::PERM) = "PERM"


struct PERMResult <: SimulationResult
    logZ::Vector{Float64}
    tour_weights::Matrix{Float64}
    num_samples::Vector{Int}
end

log_marginal(result::PERMResult) = result.logZ

# This is the main routine to compute the marginal probability in flatPERM.
function perm(alg::PERM, setup; Particle, inspect=identity)

    # this determines the time-interfaces at which we enrich or prune
    dtimes = discrete_times(setup.ensemble)

    # ln Z[n] Accumulated weight per length n. This will be the log marginal probability ln P(x)
    logZ = fill(-Inf, length(dtimes))
    tour_weights = fill(-Inf, length(dtimes), alg.num_chains)

    # H[n] counts the number of times every length n has been visited
    H = zeros(length(dtimes))

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

                # --- Update estimators ---
                logZ[n] = logaddexp(logW, logZ[n])
                tour_weights[n, S] = logaddexp(logW, tour_weights[n, S])
                H[n] += 1

                # --- Compute thresholds ---
                logW_target = logZ[n] - log(S)
                r = exp(logW - logW_target)

                n += 1 # we increased the size through propagation

                # --- Enrichment ---
                if r > 1.0
                    k = floor(r + rand())
                    logW -= log(k) # W <- W/k

                    for i = 2:k
                        push!(stack, (clone(p, setup), logW, n))
                    end
                # --- Pruning ---
                else
                    if rand() > r
                        break # prune
                    else
                        logW = logW_target # boost survivor
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
    logZ[1] = 0.0
    H[1] = alg.num_chains

    PERMResult(logZ, tour_weights, H)
end

function simulate(algorithm::PERM, initial, system; rng=Random.default_rng(), kwargs...)
    setup = Setup(initial, system, rng)
    perm(algorithm, setup; kwargs...)
end

end # module