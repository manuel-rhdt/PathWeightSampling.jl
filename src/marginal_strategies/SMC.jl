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
    weights = zeros(nparticles, length(dtimes))
    particle_indices = collect(1:nparticles)
    for (i, tspan) in enumerate(zip(dtimes[begin:end - 1], dtimes[begin + 1:end]))
        for j in eachindex(particle_bag)
            particle_bag[j] = propagate(particle_bag[j], tspan, setup)
            weights[j, i + 1] = weight(particle_bag[j])
        end

        if (i + 1) == lastindex(weights, 2)
            break
        end

        prob_weights = StatsBase.fweights(exp.(weights[:,i + 1] .- maximum(weights[:,i + 1])))

        # We only resample if the effective sample size becomes smaller than 1/2 the number of particles
        effective_sample_size = 1/sum((prob_weights ./ sum(prob_weights)) .^ 2)
        if effective_sample_size < nparticles / 2
            # sample parent indices
            parent_indices = systematic_sample(prob_weights)
            # parent_indices = StatsBase.sample(particle_indices, prob_weights, nparticles)

            particle_bag = map(parent_indices) do k
                new_particle(particle_bag[k], setup)
            end
        end
    end

    inspect(particle_bag)
    weights
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
    samples::Matrix{Float64}
end

log_marginal(result::SMCResult) = cumsum(vec(logmeanexp(result.samples, dims=1)))

function simulate(algorithm::SMCEstimate, initial, system; kwargs...)
    setup = Setup(initial, system)
    weights = sample(algorithm.num_particles, system.dtimes, setup; kwargs...)
    SMCResult(weights)
end
