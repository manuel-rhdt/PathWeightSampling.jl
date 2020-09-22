using Statistics
using FastGaussQuadrature
using LinearAlgebra

mutable struct MetropolisSampler{S}
    burn_in::Int
    skip::Int
    current_energy::Float64
    state::S
end

function Base.iterate(iter::MetropolisSampler, state=nothing)    
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

# parallel Monte-Carlo computation of the marginal probability for the given configuration
function log_marginal(initial::StochasticConfiguration, num_samples::Int, integration_nodes::Int)
    # Generate the array of θ values for which we want to simulate the system.
    # We use Gauss-Legendre quadrature which predetermines the choice of θ.
    nodes, weights = gausslegendre(integration_nodes)
    θrange = 0.5 .* nodes .+ 0.5

    energies = Array{Float64}(undef, num_samples, length(θrange))
    accept = Array{Float64}(undef, num_samples, length(θrange))
    Threads.@threads for i in eachindex(θrange)
        init = with_interaction(initial, θrange[i])
        samples, acceptance = generate_mcmc_samples(init, 50, num_samples)
        for j in eachindex(samples)
            energies[j, i] = energy(samples[j], θ=1.0)
        end
    end

    # Perform the quadrature integral. The factor 0.5 comes from rescaling the integration limits
    # from [-1,1] to [0,1].
    dot(weights, 0.5 .* vec(mean(energies, dims=1)))
end

function marginal_entropy(sn::ReactionSystem, rn::ReactionSystem, num_configurations::Int, num_samples::Int, integration_nodes::Int)
    generator = configuration_generator(sn, rn)
    result = zeros(Float64, num_configurations)
    for i in 1:num_configurations
        conf = generate_configuration(generator)
        result[i] = @time log_marginal(conf, num_samples, integration_nodes)
    end

    result
end


function conditional_entropy(sn::ReactionSystem, rn::ReactionSystem, num_configurations::Int)
    generator = configuration_generator(sn, rn)
    result = zeros(Float64, num_configurations)
    Threads.@threads for i in 1:num_configurations
        conf = generate_configuration(generator)
        result[i] = -energy(conf, θ=1.0)
    end

    result
end
