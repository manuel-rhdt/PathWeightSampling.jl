using StatsBase

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
    u_end, weight = propagate(setup.configuration, setup.ensemble, tspan)
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
        prob_weights = fweights(exp.(weights[:,i] .- maximum(weights[:,i])))
        parent_indices = StatsBase.sample(particle_indices, prob_weights, nparticles)

        particle_bag = map(parent_indices) do i
            new_particle(particle_bag[i], setup)
        end
    end

    weights
end

import GaussianMcmc: MarginalEnsemble, marginal_configuration, generate_configuration, chemotaxis_system, propagate
system = chemotaxis_system()
conf = marginal_configuration(generate_configuration(system))
ens = MarginalEnsemble(system)

setup = Setup(conf, ens)
p = new_particle(setup)
propagate!(p, (0.5, 1.0), setup)

propagate(conf, ens, (0.5, 0.6))

sample(50, 0.0:0.1:2.0, setup)
