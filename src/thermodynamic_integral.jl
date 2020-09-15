mutable struct MetropolisSampler{S<:SystemConfiguration}
    burn_in::UInt64
    skip::UInt64
    current_energy::Float64
    state::S
end

function Base.iterate(iter::MetropolisSampler, state = nothing)    
    accepted = 0
    rejected = 0
    
    new_conf = deepcopy(iter.state)
    
    while true
        propose!(new_conf, iter.state)
        new_energy = energy(new_conf)
        
        if rand() < exp(iter.current_energy - new_energy)
            accepted += 1
            iter.current_energy = new_energy
            iter.state = deepcopy(new_conf)
        else
            rejected += 1
        end
        
        if (accepted + rejected) == iter.skip + 1
            acceptance_rate = accepted / (rejected + accepted)
            return (iter.state, acceptance_rate), nothing
        end
    end
end