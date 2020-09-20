mutable struct MetropolisSampler{S}
    burn_in::Int
    skip::Int
    current_energy::Float64
    state::S
end

function Base.iterate(iter::MetropolisSampler, state = nothing)    
    accepted = 0
    rejected = 0
    
    new_conf = copy(iter.state)
    
    while true
        propose!(new_conf, iter.state)
        new_energy = energy(new_conf)
        
        if rand() < exp(iter.current_energy - new_energy)
            accepted += 1
            iter.current_energy = new_energy
            iter.state = copy(new_conf)
        else
            rejected += 1
        end
        
        if (accepted + rejected) == iter.skip + 1
            acceptance_rate = accepted / (rejected + accepted)
            return (iter.state, acceptance_rate), nothing
        end
    end
end

function generate_mcmc_samples(initial::State, skip::Int, num_samples::Int) where State
    sampler = MetropolisSampler(0, skip, energy(initial), initial)

    samples = Vector{State}(undef, num_samples)
    acceptance = zeros(num_samples)
    for (index, (sample, rate)) in Iterators.enumerate(Iterators.take(sampler, num_samples))
        samples[index] = sample
        acceptance[index] = rate
    end

    samples, acceptance
end

test_conf = StochasticConfiguration(sn, dist, response, signal, 1.0)
sampler = MetropolisSampler(0, 10000, energy(test_conf), test_conf)

samples, acceptance = generate_mcmc_samples(test_conf, 100, 1000)
p = plot([energy(s, θ=1.0) for s in samples])

using Statistics
using FastGaussQuadrature
using LinearAlgebra

num_samples = 500
nodes, weights = gausslegendre(16)
θrange = 0.5 .* nodes .+ 0.5
initial = generate_initial_configuration(sn, rn)

energies = Array{Float64}(undef, num_samples, length(θrange))
accept = Array{Float64}(undef, num_samples, length(θrange))
Threads.@threads for i in eachindex(θrange)
    init = with_interaction(initial, θrange[i])
    local samples, acceptance = generate_mcmc_samples(init, 100, num_samples)
    for j in eachindex(samples)
        energies[j, i] = energy(samples[j], θ=1.0)
    end
    accept[:, i] = acceptance
end

histogram(energies)
plot(accept)

plot(θrange, mean(energies, dims=1)', yerr=std(energies, dims=1)' ./ sqrt(size(energies, 1)))

dot(weights, 0.5 .* vec(mean(energies, dims=1)))

function iterate_plot()
    (new_state, rate), _ = iterate(sampler)
    @show rate, energy(new_state, θ=1.0)
    p = plot(new_state.signal)
    plot!(p, test_conf.signal)
    plot!(p, new_state.response)
end

iterate_plot()