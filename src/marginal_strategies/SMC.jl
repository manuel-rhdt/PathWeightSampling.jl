import StatsBase

# Each particle represents an independently evolving trajectory.
struct JumpParticle{uType}
    u::uType
    weight::Float64
end

function JumpParticle(setup)
    u = sample_initial_condition(setup.ensemble)
    w = initial_log_likelihood(setup.ensemble, u, setup.configuration.x_traj)
    JumpParticle(u, w)
end

function JumpParticle(parent::JumpParticle, setup)
    JumpParticle(copy(parent.u), 0.0)
end

function propagate(p::JumpParticle, tspan::Tuple{T,T}, setup) where {T}
    u_end, weight = propagate(setup.configuration, setup.ensemble, p.u, tspan)
    JumpParticle(u_end, weight)
end

# While this implementation of a particle is slower than the one above
# it keeps track of the genealogy of a particle (i.e. its parent, grandparent, etc.).
# This is useful for debugging and for collecting statistics.
mutable struct ParentTrackingParticle{Inner}
    p::Inner
    parent::Union{ParentTrackingParticle{Inner},Nothing}
end

struct Setup{Configuration,Ensemble}
    configuration::Configuration
    ensemble::Ensemble
end

function ParentTrackingParticle{Inner}(setup) where {Inner}
    ParentTrackingParticle(Inner(setup), nothing)
end

function ParentTrackingParticle(parent::ParentTrackingParticle{Inner}, setup) where {Inner}
    ParentTrackingParticle(Inner(parent.p, setup), parent)
end

function propagate(p::ParentTrackingParticle, tspan::Tuple{T,T}, setup) where {T}
    new_p = propagate(p.p, tspan, setup)
    p.p = new_p
    p
end

weight(p::JumpParticle) = p.weight
weight(p::ParentTrackingParticle) = weight(p.p)

# This is the main routine to compute the marginal probability in RR-PWS.
function sample(setup, nparticles; inspect=Base.identity, new_particle=JumpParticle)
    particle_bag = [new_particle(setup) for i = 1:nparticles]
    weights = zeros(nparticles)
    dtimes = discrete_times(setup)

    log_marginal_estimate = zeros(length(dtimes))

    # First, handle initial condition
    for j in eachindex(particle_bag)
        weights[j] += weight(particle_bag[j])
    end
    log_marginal_estimate[1] = logmeanexp(weights)

    # At each time we do the following steps
    # 1) propagate all particles forwards in time and compute their weights
    # 2) update the log_marginal_estimate
    # 3) check whether we should resample the particle bag, and resample if needed
    for (i, tspan) in enumerate(zip(dtimes[begin:end-1], dtimes[begin+1:end]))

        # PROPAGATE
        for j in eachindex(particle_bag)
            particle_bag[j] = propagate(particle_bag[j], tspan, setup)
            weights[j] += weight(particle_bag[j])
        end

        inspect(particle_bag) #< useful for collecting statistics

        # UPDATE ESTIMATE
        log_marginal_estimate[i+1] += logmeanexp(weights)

        if (i + 1) == lastindex(dtimes)
            break
        end

        # RESAMPLE IF NEEDED
        prob_weights = StatsBase.weights(exp.(weights .- maximum(weights)))

        # We only resample if the effective sample size becomes smaller than 1/2 the number of particles
        effective_sample_size = 1 / sum(p -> (p / sum(prob_weights))^2, prob_weights)
        if effective_sample_size < nparticles / 2
            if effective_sample_size <= 5
                @warn "Small effective sample size" i tspan effective_sample_size
            end
            # sample parent indices
            parent_indices = systematic_sample(prob_weights)

            particle_bag = map(parent_indices) do k
                new_particle(particle_bag[k], setup)
            end
            weights .= 0.0
            log_marginal_estimate[i+1:end] .= log_marginal_estimate[i+1]
        end
    end

    log_marginal_estimate
end

"""
    systematic_sample(weights)::Vector{Int}

Take random samples from `1:len(weights)` using weights given by the corresponding values
in the `weights` array.

The number of samples returned is equal to the length of `weights`.
"""
function systematic_sample(weights; N=length(weights))
    inc = 1 / N
    x = inc * rand()
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
struct SMCEstimate
    num_particles::Int
end

name(x::SMCEstimate) = "SMC"

struct SMCResult <: SimulationResult
    log_marginal_estimate::Vector{Float64}
end


log_marginal(result::SMCResult) = result.log_marginal_estimate


function simulate(algorithm::SMCEstimate, initial, system; kwargs...)
    setup = Setup(initial, system)
    log_marginal_estimate = sample(setup, algorithm.num_particles; kwargs...)
    SMCResult(log_marginal_estimate)
end
