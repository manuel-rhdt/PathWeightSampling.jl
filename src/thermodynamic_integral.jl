using Statistics
using FastGaussQuadrature
using LinearAlgebra
using DataFrames

mutable struct MetropolisSampler{S, Sys}
    skip::Int
    current_energy::Float64
    state::S
    system::Sys
end

Base.iterate(sampler::MetropolisSampler) = iterate(sampler, deepcopy(sampler.state))

function Base.iterate(sampler::MetropolisSampler{S, Sys}, new_state::S) where {S, Sys}    
    accepted = 0
    rejected = 0

    while true
        propose!(new_state, sampler.state, sampler.system)
        new_energy = energy(new_state, sampler.system)
        
        if rand() < exp(sampler.current_energy - new_energy)
            accepted += 1
            sampler.current_energy = new_energy
            # simple variable swap
            tmp = new_state
            new_state = sampler.state
            sampler.state = tmp
        else
            rejected += 1
        end
        
        if (accepted + rejected) == sampler.skip + 1
            acceptance_rate = accepted / (rejected + accepted)
            return (deepcopy(sampler.state), acceptance_rate), new_state
        end
    end
end

function generate_mcmc_samples(initial::State, system, skip::Int, num_samples::Int) where State
    sampler = MetropolisSampler(skip, energy(initial, system), initial, system)

    samples = Vector{State}(undef, num_samples)
    acceptance = zeros(num_samples)
    for (index, (sample, rate)) in Iterators.enumerate(Iterators.take(sampler, num_samples))
        samples[index] = sample
        acceptance[index] = rate
    end

    samples, acceptance
end

# parallel Monte-Carlo computation of the marginal probability for the given configuration
function log_marginal(initial::Trajectory, system::StochasticSystem, num_samples::Int, integration_nodes::Int, skip_samples::Int)
    # Generate the array of θ values for which we want to simulate the system.
    # We use Gauss-Legendre quadrature which predetermines the choice of θ.
    nodes, weights = gausslegendre(integration_nodes)
    θrange = 0.5 .* nodes .+ 0.5

    energies = Array{Float64}(undef, num_samples, length(θrange))
    accept = Array{Float64}(undef, num_samples, length(θrange))
    for i in eachindex(θrange)
        system.θ = θrange[i]
        samples, acceptance = generate_mcmc_samples(initial, system, skip_samples, num_samples)
        for j in eachindex(samples)
            energies[j, i] = energy(samples[j], system, θ=1.0)
            accept[j, i] = acceptance[j]
        end
    end

    # Perform the quadrature integral. The factor 0.5 comes from rescaling the integration limits
    # from [-1,1] to [0,1].
    dot(weights, 0.5 .* vec(mean(energies, dims=1))), mean(accept)
end

function marginal_entropy(
        sn::ReactionSystem, 
        rn::ReactionSystem; 
        num_responses::Int=1,
        num_samples::Int=1000, 
        skip_samples::Int=50,
        integration_nodes::Int=16,
        duration::Float64=500.0
    )
    generator = configuration_generator(sn, rn)
    result = DataFrame(
        Sample=zeros(Float64, num_responses), 
        Acceptance=zeros(Float64, num_responses), 
        TimeElapsed=zeros(Float64, num_responses), 
        GcTime=zeros(Float64, num_responses)
    )
    #Threads.@threads 
    for i in 1:num_responses
        (system, initial) = generate_configuration(generator; duration=duration)
        lm = @timed log_marginal(initial, system, num_samples, integration_nodes, skip_samples)
        (val, acc) = lm.value
        elapsed = lm.time
        gctime = lm.gctime

        result.Sample[i] = val
        result.Acceptance[i] = acc
        result.TimeElapsed[i] = elapsed
        result.GcTime[i] = gctime

        println(NamedTuple(result[i, :]))
    end

    result
end


function conditional_entropy(sn::ReactionSystem, rn::ReactionSystem; num_responses::Int=1, duration::Float64=500.0)
    generator = configuration_generator(sn, rn)
    result = zeros(Float64, num_responses)
    #Threads.@threads 
    for i in 1:num_responses
        (system, initial) = generate_configuration(generator, duration=duration)
        result[i] = energy(initial, system)
    end

    DataFrame(
        Sample=result
    )
end
