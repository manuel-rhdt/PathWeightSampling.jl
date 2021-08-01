import StatsBase

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

function sample(nparticles, dtimes, setup; inspect=Base.identity, new_particle=JumpParticle)
    particle_bag = [new_particle(setup) for i = 1:nparticles]
    weights = zeros(nparticles)
    log_marginal_estimate = zeros(length(dtimes))
    for (i, tspan) in enumerate(zip(dtimes[begin:end - 1], dtimes[begin + 1:end]))
        for j in eachindex(particle_bag)
            particle_bag[j] = propagate(particle_bag[j], tspan, setup)
            weights[j] += weight(particle_bag[j])
        end

        log_marginal_estimate[i+1] += logmeanexp(weights)

        if (i + 1) == lastindex(dtimes)
            break
        end

        prob_weights = StatsBase.fweights(exp.(weights .- maximum(weights)))

        # We only resample if the effective sample size becomes smaller than 1/2 the number of particles
        effective_sample_size = 1/sum(p -> (p / sum(prob_weights)) ^ 2, prob_weights)
        if effective_sample_size < nparticles / 2
            # sample parent indices
            parent_indices = systematic_sample(prob_weights)
            # parent_indices = StatsBase.sample(particle_indices, prob_weights, nparticles)

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
