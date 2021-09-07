import StatsBase

# Each particle represents an independently evolving trajectory.
struct JumpParticle{uType}
    u::uType
    weight::Float64
end

function JumpParticle(setup)
    u = setup.ensemble.jump_problem.prob.u0
    JumpParticle(u, 0.0)
end

function JumpParticle(parent::JumpParticle, setup)
    JumpParticle(copy(parent.u), 0.0)
end

function propagate(p::JumpParticle, tspan::Tuple{T,T}, setup) where T
    u_end, weight = propagate(setup.configuration, setup.ensemble, p.u, tspan)
    JumpParticle(u_end, weight)
end

# While this implementation of a particle is slower than the one above
# it keeps track of the genealogy of a particle (i.e. its parent, grandparent, etc.).
# This is useful for debugging and for collecting statistics.
mutable struct JumpParticleSlow{uType}
    u::uType
    weight::Float64
    parent::Union{JumpParticleSlow{uType},Nothing}

    function JumpParticleSlow(setup)
        u = setup.ensemble.jump_problem.prob.u0
        new{typeof(u)}(u, 0.0, nothing)
    end

    function JumpParticleSlow(parent, setup)
        new{typeof(parent.u)}(copy(parent.u), 0.0, parent)
    end
end

struct Setup{Configuration,Ensemble}
    configuration::Configuration
    ensemble::Ensemble
end

function propagate(p::JumpParticleSlow, tspan::Tuple{T,T}, setup) where T
    u_end, weight = propagate(setup.configuration, setup.ensemble, p.u, tspan)
    p.u = u_end
    p.weight = weight
    p
end

weight(p::JumpParticle) = p.weight
weight(p::JumpParticleSlow) = p.weight

# This is the main algorithm of SMC-PWS.
function sample(nparticles, dtimes, setup; inspect=Base.identity, new_particle=JumpParticle)
    particle_bag = [new_particle(setup) for i = 1:nparticles]
    weights = zeros(nparticles)
    log_marginal_estimate = zeros(length(dtimes))

    # At each time we do the following steps
    # 1) propagate all particles forwards in time and compute their weights
    # 2) update the log_marginal_estimate
    # 3) check whether we should resample the particle bag, and resample if needed
    for (i, tspan) in enumerate(zip(dtimes[begin:end - 1], dtimes[begin + 1:end]))
        for j in eachindex(particle_bag)
            particle_bag[j] = propagate(particle_bag[j], tspan, setup)
            weights[j] += weight(particle_bag[j])
        end

        log_marginal_estimate[i+1] += logmeanexp(weights)

        if (i + 1) == lastindex(dtimes)
            break
        end

        prob_weights = StatsBase.weights(exp.(weights .- maximum(weights)))

        # We only resample if the effective sample size becomes smaller than 1/2 the number of particles
        effective_sample_size = 1/sum(p -> (p / sum(prob_weights)) ^ 2, prob_weights)
        if effective_sample_size < nparticles / 2
            # sample parent indices
            parent_indices = systematic_sample(prob_weights)

            particle_bag = map(parent_indices) do k
                new_particle(particle_bag[k], setup)
            end
            weights .= 0.0
            log_marginal_estimate[i+1:end] .= log_marginal_estimate[i+1]
        end
    end

    inspect(particle_bag)
    log_marginal_estimate
end

"""
    systematic_sample(weights)::Vector{Int}

Take random samples from `1:len(weights)` using weights given by the corresponding values
in the `weights` array.

The number of samples returned is equal to the length of `weights`.
"""
function systematic_sample(weights)
    N = length(weights)
    inc = 1/N
    x = inc * rand()
    j=1
    y = weights[j] / sum(weights)
    result = zeros(Int, N)
    for i=1:N
        while y < x
            j += 1
            y += weights[j] / sum(weights)
        end
        result[i] = j
        x += inc
    end
    result
end

struct SMCEstimate
    num_particles::Int
end

name(x::SMCEstimate) = "SMC"

struct SMCResult <: SimulationResult
    log_marginal_estimate::Vector{Float64}
end

# log_marginal(result::SMCResult) = cumsum(vec(logmeanexp(result.samples, dims=1)))
log_marginal(result::SMCResult) = result.log_marginal_estimate


function simulate(algorithm::SMCEstimate, initial, system; kwargs...)
    setup = Setup(initial, system)
    log_marginal_estimate = sample(algorithm.num_particles, system.dtimes, setup; kwargs...)
    SMCResult(log_marginal_estimate)
end
