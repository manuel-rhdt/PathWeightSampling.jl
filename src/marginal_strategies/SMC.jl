import StatsBase

mutable struct JumpParticle{uType}
    u::uType
    weight::Float64
end

struct Setup{Configuration,Ensemble}
    configuration::Configuration
    ensemble::Ensemble
end

function new_particle(setup)
    u = setup.ensemble.jump_problem.prob.u0
    JumpParticle(u, 0.0)
end

function new_particle(parent, setup)
    u = parent.u
    JumpParticle(u, 0.0)
end

function propagate!(p::JumpParticle, tspan::Tuple{T,T}, setup) where T
    u_end, weight = propagate(setup.configuration, setup.ensemble, p.u, tspan)
    p.u = u_end
    p.weight += weight
    p
end

weight(p::JumpParticle) = p.weight

function sample(nparticles, dtimes, setup)
    particle_bag = [new_particle(setup) for i = 1:nparticles]
    weights = zeros(nparticles, length(dtimes) - 1)
    particle_indices = collect(1:nparticles)
    for (i, tspan) in enumerate(zip(dtimes[begin:end - 1], dtimes[begin + 1:end]))
        for (j, particle) in enumerate(particle_bag)
            propagate!(particle, tspan, setup)
            weights[j, i] = weight(particle)
        end

        if i == lastindex(weights, 2)
            break
        end

        # sample parent indices
        prob_weights = StatsBase.fweights(exp.(weights[:,i] .- maximum(weights[:,i])))
        parent_indices = StatsBase.sample(particle_indices, prob_weights, nparticles)

        particle_bag = map(parent_indices) do i
            new_particle(particle_bag[i], setup)
        end
    end

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

function simulate(algorithm::SMCEstimate, initial, system)
    setup = Setup(initial, system)
    weights = sample(algorithm.num_particles, system.dtimes, setup)
    SMCResult(weights)
end
