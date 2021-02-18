import StatsBase

mutable struct JumpParticle{uType}
    u::uType
    weight::Float64
end

function new_particle{JumpParticle}(setup)
    u = setup.ensemble.jump_problem.prob.u0
    JumpParticle(u, 0.0, nothing)
end

function new_particle{JumpParticle}(parent, setup)
    JumpParticle(parent.u, 0.0, parent)
end

function propagate!(p::JumpParticle, tspan::Tuple{T,T}, setup) where T
    u_end, weight = propagate(setup.configuration, setup.ensemble, p.u, tspan)
    p.u = u_end
    p.weight = weight
    p
end

mutable struct JumpParticleSlow{uType}
    u::uType
    weight::Float64
    parent::Union{JumpParticle{uType},Nothing}
end

struct Setup{Configuration,Ensemble}
    configuration::Configuration
    ensemble::Ensemble
end

function new_particle{JumpParticleSlow}(setup)
    u = setup.ensemble.jump_problem.prob.u0
    JumpParticle(u, 0.0, nothing)
end

function new_particle{JumpParticleSlow}(parent, setup)
    JumpParticle(parent.u, 0.0, parent)
end

function propagate!(p::JumpParticleSlow, tspan::Tuple{T,T}, setup) where T
    u_end, weight = propagate(setup.configuration, setup.ensemble, p.u, tspan)
    p.u = u_end
    p.weight = weight
    p
end

weight(p::JumpParticleSlow) = p.weight

function sample(nparticles, dtimes, setup; inspect=Base.identity, ptype=JumpParticle)
    particle_bag = [new_particle{ptype}(setup) for i = 1:nparticles]
    weights = zeros(nparticles, length(dtimes))
    particle_indices = collect(1:nparticles)
    for (i, tspan) in enumerate(zip(dtimes[begin:end - 1], dtimes[begin + 1:end]))
        for (j, particle) in enumerate(particle_bag)
            propagate!(particle, tspan, setup)
            weights[j, i + 1] = weight(particle)
        end

        if (i + 1) == lastindex(weights, 2)
            break
        end

        # sample parent indices
        prob_weights = StatsBase.fweights(exp.(weights[:,i + 1] .- maximum(weights[:,i + 1])))
        parent_indices = StatsBase.sample(particle_indices, prob_weights, nparticles)

        if i == 100
            @show sort(weights[:,i + 1])
        end

        particle_bag = map(parent_indices) do k
            new_particle(particle_bag[k], setup)
        end
    end

    inspect(particle_bag)
    weights
end


struct SMCEstimate
    num_particles::Int
end

name(x::SMCEstimate) = "SMC"

struct SMCResult <: SimulationResult
    samples::Matrix{Float64}
end

log_marginal(result::SMCResult) = cumsum(vec(logmeanexp(result.samples, dims=1)))

function simulate(algorithm::SMCEstimate, initial, system; inspect=Base.identity)
    setup = Setup(initial, system)
    weights = sample(algorithm.num_particles, system.dtimes, setup; inspect)
    SMCResult(weights)
end
