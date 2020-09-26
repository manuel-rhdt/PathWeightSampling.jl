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
            return acceptance_rate, new_state
        end
    end
end

function generate_mcmc_samples(initial::State, system, skip::Int, num_samples::Int) where State
    sampler = MetropolisSampler(skip, energy(initial, system), initial, system)

    samples = Vector{State}(undef, num_samples)
    acceptance = zeros(num_samples)
    for (index, rate) in Iterators.enumerate(Iterators.take(sampler, num_samples))
        samples[index] = deepcopy(sampler.state)
        acceptance[index] = rate
    end

    samples, acceptance
end

function annealed_importance_sampling(initial, system::StochasticSystem, skip::Int, num_samples::Int)
    weights = zeros(Float64, num_samples)
    temps = range(0, 1; length=num_samples + 1)
    acceptance = zeros(Float64, num_samples)
    acceptance[1] = 1.0

    e_prev = energy(initial, system, θ = temps[1])
    e_cur = energy(initial, system, θ = temps[2])
    weights[1] = e_prev - e_cur

    system.θ = temps[2]
    sampler = MetropolisSampler(skip, e_cur, initial, system)
    for (i, acc) in zip(2:num_samples, sampler)
        e_prev = sampler.current_energy
        e_cur = energy(sampler.state, system, θ = temps[i + 1])

        weights[i] = weights[i-1] + e_prev - e_cur
        acceptance[i] = acc

        # change the temperature for the next iteration
        sampler.system.θ = temps[i + 1]
        sampler.current_energy = e_cur
    end
    temps[2:end], weights, acceptance
end

function log_marginal(::Val{:Annealing}, initial::Trajectory, system::StochasticSystem, skip::Int, num_temps::Int, num_samples::Int)
    all_weights = zeros(Float64, num_temps, num_samples)
    for i in 1:num_samples
        signal = new_signal(initial, system)
        (temps, weights, acceptance) = annealed_importance_sampling(signal, system, skip, num_temps)
        all_weights[:, i] = weights
    end

    range(0,1,length=num_temps + 1), all_weights
end

# Monte-Carlo computation of the marginal probability for the given configuration
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
        gen::ConfigurationGenerator; 
        num_responses::Int=1,
        num_samples::Int=1000, 
        skip_samples::Int=50,
        integration_nodes::Int=16,
        duration::Float64=500.0
    )
    result = DataFrame(
        Sample=zeros(Float64, num_responses), 
        Acceptance=zeros(Float64, num_responses), 
        TimeElapsed=zeros(Float64, num_responses), 
        GcTime=zeros(Float64, num_responses)
    )
    #Threads.@threads 
    for i in 1:num_responses
        (system, initial) = generate_configuration(gen; duration=duration)
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


function conditional_entropy(gen::ConfigurationGenerator;  num_responses::Int=1, duration::Float64=500.0)
    result = zeros(Float64, num_responses)
    #Threads.@threads 
    for i in 1:num_responses
        (system, initial) = generate_configuration(gen, duration=duration)
        result[i] = energy(initial, system)
    end

    DataFrame(
        Sample=result
    )
end
