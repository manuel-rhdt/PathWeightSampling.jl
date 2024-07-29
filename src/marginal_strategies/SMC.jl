module SMC

export SMCEstimate, weight, propagate, AbstractParticle, clone

import ..PathWeightSampling: AbstractSimulationAlgorithm, SimulationResult, discrete_times,
    log_marginal, logmeanexp, name, simulate

import Random
import StatsBase


abstract type AbstractParticle end

weight(p::AbstractParticle) = 0.0
propagate(p::AbstractParticle, tspan::Tuple{T,T}, setup::Any) where {T} = error("needs to be implemented")
clone(p::AbstractParticle, setup::Any) = copy(p)

struct Setup{Configuration,Ensemble,RNG}
    configuration::Configuration
    ensemble::Ensemble
    rng::RNG
end

# This is the main routine to compute the marginal probability in RR-PWS.
function sample(
    setup::Any,
    particles::Vector{<:AbstractParticle};
    inspect=Base.identity,
    resample_threshold=length(particles) / 2,
    rng=Random.default_rng()
)
    nparticles = length(particles)
    weights = zeros(nparticles)
    dtimes = discrete_times(setup)

    log_marginal_estimate = zeros(length(dtimes))

    # First, handle initial condition
    for j in eachindex(particles)
        weights[j] += weight(particles[j])
    end
    log_marginal_estimate[1] = logmeanexp(weights)

    # At each time we do the following steps
    # 1) propagate all particles forwards in time and compute their weights
    # 2) update the log_marginal_estimate
    # 3) check whether we should resample the particle bag, and resample if needed
    for (i, tspan) in enumerate(zip(dtimes[begin:end-1], dtimes[begin+1:end]))

        # PROPAGATE
        for j in eachindex(particles)
            particles[j] = propagate(particles[j], tspan, setup)
            weights[j] += weight(particles[j])
        end

        inspect(particles) #< useful for collecting statistics

        # UPDATE ESTIMATE
        log_marginal_estimate[i+1] += logmeanexp(weights)

        if isnan(log_marginal_estimate[i+1])
            print(weights)
            error("unexpected NaN")
        end

        if (i + 1) >= lastindex(dtimes)
            break
        end

        # RESAMPLE IF NEEDED
        prob_weights = StatsBase.weights(exp.(weights .- maximum(weights)))

        # We only resample if the effective sample size becomes smaller than the threshold
        effective_sample_size = 1 / sum(p -> (p / sum(prob_weights))^2, prob_weights)
        if effective_sample_size < resample_threshold
            @debug "Resample" i tspan effective_sample_size
            if effective_sample_size <= 5
                @debug "Small effective sample size" i tspan effective_sample_size
            end
            # sample parent indices
            parent_indices = systematic_sample(rng, prob_weights)

            new_particles = similar(particles)
            for (i, k) in enumerate(parent_indices)
                new_particles[i] = clone(particles[k], setup)
            end
            particles = new_particles

            weights .= 0.0
            log_marginal_estimate[i+1:end] .= log_marginal_estimate[i+1]
        end
    end

    log_marginal_estimate
end

"""
    systematic_sample(weights[; N])::Vector{Int}

Take random samples from `1:len(weights)` using weights given by the corresponding values
in the `weights` array.

The number of samples returned is by default equal to the length of `weights`. If a different
number of samples is required, set `N` to the desired number.
"""
function systematic_sample(rng, weights; N=length(weights))
    inc = 1 / N
    x = inc * rand(rng)
    j = 1
    y = weights[j] / sum(weights)
    result = zeros(Int, N)
    for i = 1:N
        while y < x
            j += 1
            y += weights[j] / sum(weights)
        end
        result[i] = j
        x += inc
    end
    result
end

"""
    SMCEstimate(num_particles::Int)

Compute the marginal trajectory probability using a Sequential Monte Carlo (SMC)
algorithm, more specifically using a [particle filter](https://en.wikipedia.org/wiki/Particle_filter).

In a particle filter, `num_particles` trajectories are propagated in parallel. At regular
intervals, the current set of parallel trajectories undergoes a resampling step where
some of the trajectories get eliminated and others duplicated, depending on their accumulated
likelihood. This resampling ensures that we don't sample trajectories that contribute only 
very little to the marginalization integral.

Due to the periodic resampling of the trajectories, this method works much better than the 
`DirectMCEstimate` for long trajectories.
"""
struct SMCEstimate <: AbstractSimulationAlgorithm
    num_particles::Int
end

name(x::SMCEstimate) = "SMC-$(x.num_particles)"

struct SMCResult <: SimulationResult
    log_marginal_estimate::Vector{Float64}
end


log_marginal(result::SMCResult) = result.log_marginal_estimate


function simulate(algorithm::SMCEstimate, initial, system; rng=Random.default_rng(), new_particle, kwargs...)
    setup = Setup(initial, system, rng)
    particles = [new_particle(setup) for i = 1:algorithm.num_particles]
    log_marginal_estimate = sample(setup, particles; rng=rng, kwargs...)
    SMCResult(log_marginal_estimate)
end

end # module