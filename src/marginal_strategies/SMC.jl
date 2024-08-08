module SMC

export SMCEstimate, weight, propagate, AbstractParticle, clone

import ..PathWeightSampling: AbstractSimulationAlgorithm, SimulationResult, discrete_times,
    log_marginal, to_dataframe, logmeanexp, name, simulate

import Random
import StatsBase
using DataFrames

abstract type AbstractParticle end

weight(p::AbstractParticle) = 0.0
spawn(::Type{P}, setup::Any) where P = error!("needs to be implemented")
propagate!(p::AbstractParticle, tspan::Tuple{T,T}, setup::Any) where {T} = error("needs to be implemented")
clone(parent::P, setup::Any) where P <: AbstractParticle = clone_from!(spawn(P, setup), parent, setup)
clone_from!(child::AbstractParticle, parent::AbstractParticle, setup::Any) = error("needs to be implemented")

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
    dtimes = discrete_times(setup.ensemble)

    log_marginal_estimate = zeros(length(dtimes))
    effective_sample_sizes = zeros(length(dtimes))
    effective_sample_sizes[1] = nparticles

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
            particles[j] = propagate!(particles[j], tspan, setup)
            weights[j] += weight(particles[j])
        end

        inspect(particles) #< useful for collecting statistics

        # UPDATE ESTIMATE
        log_marginal_estimate[i+1] += logmeanexp(weights)

        prob_weights = StatsBase.weights(exp.(weights .- maximum(weights)))
        # We only resample if the effective sample size becomes smaller than the threshold
        effective_sample_size = 1 / sum(p -> (p / sum(prob_weights))^2, prob_weights)
        effective_sample_sizes[i+1] = effective_sample_size

        if (i + 1) >= lastindex(dtimes)
            break
        end

        # RESAMPLE IF NEEDED
        if effective_sample_size < resample_threshold
            @debug "Resample" i tspan effective_sample_size
            if effective_sample_size <= 5
                @debug "Small effective sample size" i tspan effective_sample_size
            end
            # sample parent indices
            parent_indices = sort!(systematic_sample(rng, prob_weights))

            overwritten = BitSet()
            for (child_index, parent_index) in zip(eachindex(particles), parent_indices)
                if child_index < parent_index
                    @assert !in(parent_index, overwritten)
                    clone_from!(particles[child_index], particles[parent_index], setup)
                    push!(overwritten, child_index)
                end
            end
            for (child_index, parent_index) in zip(reverse(eachindex(particles)), reverse!(parent_indices))
                if child_index > parent_index
                    @assert !in(parent_index, overwritten)
                    clone_from!(particles[child_index], particles[parent_index], setup)
                    push!(overwritten, child_index)
                end
            end
            @assert union(overwritten, parent_indices) == BitSet(eachindex(particles))

            weights .= 0.0
            log_marginal_estimate[i+1:end] .= log_marginal_estimate[i+1]
        end
    end

    log_marginal_estimate, effective_sample_sizes
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
    time::Vector{Float64}
    log_marginal_estimate::Vector{Float64}
    effective_sample_sizes::Vector{Float64}
end

log_marginal(result::SMCResult) = result.log_marginal_estimate

function to_dataframe(result::SMCResult)
    DataFrame(:time => result.time, :log_marginal => result.log_marginal_estimate, :ESS => result.effective_sample_sizes)
end

function simulate(algorithm::SMCEstimate, initial, system; rng=Random.default_rng(), Particle, kwargs...)
    setup = Setup(initial, system, rng)
    particles = [spawn(Particle, setup) for i = 1:algorithm.num_particles]
    log_marginal_estimate, effective_sample_sizes = sample(setup, particles; rng=rng, kwargs...)
    SMCResult(Vector(discrete_times(system)), log_marginal_estimate, effective_sample_sizes)
end

end # module