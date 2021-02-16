
mutable struct JumpParticle

end

function propagate!(p::JumpParticle, tspan::Tuple{T, T}, system) where T
    # set up a Gillespie simulation for the tspan

    weight = log_likelihood(system.configuration, system.ensemble, tspan)

    # extract end position
    # compute P(s,x)/P(s) and store in weight field
end

function sample()
    for particle in particle_bag
        propagate!(particle)
        weight(particle)
    end

    # sample parent indices

    particle_bag = map(1:parent_indices) do i
        np = new_particle(particle_bag[i])
    end
end

