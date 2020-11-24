
import Random
import Base.Iterators

abstract type MarkovChain end

function accept(::MarkovChain)
    nothing
end

function reject(::MarkovChain)
    nothing
end

propose(old_state, chain::MarkovChain) = propose!(copy(old_state), old_state, chain)

mutable struct MetropolisSampler{S,Sys}
    burn_in::Int
    subsample::Int
    current_energy::Float64
    state::S
    chain::Sys
end

MetropolisSampler(state, chain; burn_in::Int=0, subsample::Int=0) = MetropolisSampler(burn_in, subsample, energy(state, chain), state, chain)

function set_state(sampler::MetropolisSampler{S}, state::S) where {S}
    sampler.current_energy = energy(state, sampler.chain)
    sampler.state = state
end

Base.IteratorSize(::Type{<:MetropolisSampler}) = Base.IsInfinite()

Base.iterate(sampler::MetropolisSampler) = iterate(sampler, copy(sampler.state))

function Base.iterate(sampler::MetropolisSampler{S,Sys}, new_state::S) where {S,Sys}    
    accepted = 0
    rejected = 0

    while true
        propose!(new_state, sampler.state, sampler.chain)
        new_energy = energy(new_state, sampler.chain)
        
        # metropolis acceptance criterion
        if Random.randexp() >= new_energy - sampler.current_energy
            accept(sampler.chain)
            accepted += 1
            sampler.current_energy = new_energy
            # simple variable swap (sampler.state <--> new_state)
            tmp = new_state
            new_state = sampler.state
            sampler.state = tmp
        else
            reject(sampler.chain)
            rejected += 1
        end
        
        if (accepted + rejected) > max(sampler.burn_in, sampler.subsample)
            sampler.burn_in = 0
            return accepted, new_state
        end
    end
end


function sample(sampler::MetropolisSampler)
    iterate(sampler)
    copy(sampler.state)
end

function sample(sampler::MetropolisSampler, num_samples::Integer)
    map(Iterators.take(sampler, num_samples)) do _
        copy(sampler.state)
    end
end

function sample(f, sampler::MetropolisSampler, num_samples::Integer)
    map(Iterators.take(sampler, num_samples)) do _
        f(sampler.state)
    end
end

