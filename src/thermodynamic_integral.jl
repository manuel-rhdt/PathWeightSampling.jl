using Statistics
using FastGaussQuadrature
using LinearAlgebra
using DataFrames
using StatsFuns

mutable struct MetropolisSampler{S,Sys}
    skip::Int
    current_energy::Float64
    state::S
    system::Sys
end

Base.iterate(sampler::MetropolisSampler) = iterate(sampler, deepcopy(sampler.state))

function Base.iterate(sampler::MetropolisSampler{S,Sys}, new_state::S) where {S,Sys}    
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

    e_prev = energy(initial, system, θ=temps[1])
    e_cur = energy(initial, system, θ=temps[2])
    weights[1] = e_prev - e_cur

    system.θ = temps[2]
    sampler = MetropolisSampler(skip, e_cur, initial, system)
    for (i, acc) in zip(2:num_samples, sampler)
        e_prev = sampler.current_energy
        e_cur = energy(sampler.state, system, θ=temps[i + 1])

        weights[i] = weights[i - 1] + e_prev - e_cur
        acceptance[i] = acc

        # change the temperature for the next iteration
        sampler.system.θ = temps[i + 1]
        sampler.current_energy = e_cur
    end
    temps, weights, acceptance
end

struct AnnealingEstimate
    skip::Int
    num_temps::Int
    num_samples::Int
end

abstract type SimulationResult end

function write_hdf5!(group, res_array::Vector{<: SimulationResult})
end

struct AnnealingEstimationResult <: SimulationResult
    inv_temps::Vector{Float64}
    weights::Array{Float64,2}
    acceptance::Array{Float64,2}
end

log_marginal(result::AnnealingEstimationResult) =  -(logsumexp(result.weights[end, :]) - log(size(result.weights, 2)))

function write_hdf5!(group, res_array::Vector{AnnealingEstimationResult})
    inv_temps = cat((r.inv_temps for r in res_array)...; dims=2)
    weights = cat((r.weights for r in res_array)...; dims=3)
    acceptance = cat((r.acceptance for r in res_array)...; dims=3)

    group["inv_temps"] = inv_temps[:, 1]
    group["weights"] = weights
    group["acceptance"] = acceptance
    nothing
end

function simulate(algorithm::AnnealingEstimate, initial::Trajectory, system::StochasticSystem)
    all_weights = zeros(Float64, algorithm.num_temps, algorithm.num_samples)
    acc = zeros(Float64, algorithm.num_temps, algorithm.num_samples)
    inv_temps = nothing
    for i in 1:algorithm.num_samples
        signal = new_signal(initial, system)
        (temps, weights, acceptance) = annealed_importance_sampling(signal, system, algorithm.skip, algorithm.num_temps)
        inv_temps = temps
        all_weights[:, i] = weights
        acc[:, i] = acceptance
    end

    AnnealingEstimationResult(collect(inv_temps), all_weights, acc)
end

struct TIEstimate
    skip::Int
    integration_nodes::Int
    num_samples::Int
end

struct ThermodynamicIntegrationResult <: SimulationResult
    integration_weights::Vector{Float64}
    inv_temps::Vector{Float64}
    energies::Array{Float64,2}
    acceptance::Array{Float64,2}
end

function write_hdf5!(group, res_array::Vector{ThermodynamicIntegrationResult})
    inv_temps = cat((r.inv_temps for r in res_array)...; dims=2)
    integration_weights = cat((r.integration_weights for r in res_array)...; dims=2)
    energies = cat((r.energies for r in res_array)...; dims=3)
    acceptance = cat((r.acceptance for r in res_array)...; dims=3)

    group["inv_temps"] = inv_temps[:, 1]
    group["integration_weights"] = integration_weights
    group["energies"] = energies
    group["acceptance"] = acceptance
    nothing
end

# perform the quadrature integral
log_marginal(result::ThermodynamicIntegrationResult) = dot(result.integration_weights, vec(mean(result.energies, dims=1)))

# Monte-Carlo computation of the marginal probability for the given configuration
function simulate(algorithm::TIEstimate, initial::Trajectory, system::StochasticSystem)
    # Generate the array of θ values for which we want to simulate the system.
    # We use Gauss-Legendre quadrature which predetermines the choice of θ.
    nodes, weights = gausslegendre(algorithm.integration_nodes)
    θrange = 0.5 .* nodes .+ 0.5
    # The factor 0.5 comes from rescaling the integration limits from [-1,1] to [0,1].
    weights = 0.5 .* weights

    energies = Array{Float64}(undef, algorithm.num_samples, length(θrange))
    accept = Array{Float64}(undef, algorithm.num_samples, length(θrange))
    for i in eachindex(θrange)
        system.θ = θrange[i]
        samples, acceptance = generate_mcmc_samples(initial, system, algorithm.skip, algorithm.num_samples)
        for j in eachindex(samples)
            energies[j, i] = energy(samples[j], system, θ=1.0)
            accept[j, i] = acceptance[j]
        end
    end

    ThermodynamicIntegrationResult(weights, θrange, energies, accept)
end

function marginal_entropy(
        gen::ConfigurationGenerator;
        algorithm,
        num_responses::Int=1,
        duration::Float64=500.0
    )
    stats = DataFrame(TimeElapsed=zeros(Float64, num_responses), GcTime=zeros(Float64, num_responses))

    result = map(1:num_responses) do i
        (system, initial) = generate_configuration(gen; duration=duration)
        timed_result = @timed simulate(algorithm, initial, system)

        stats.TimeElapsed[i] = timed_result.time
        stats.GcTime[i] = timed_result.gctime

        timed_result.value
    end

    result, stats
end


function conditional_entropy(gen::ConfigurationGenerator;  num_responses::Int=1, duration::Float64=500.0)
    result = zeros(Float64, num_responses)
    for i in 1:num_responses
        (system, initial) = generate_configuration(gen, duration=duration)
        result[i] = energy(initial, system)
    end

    DataFrame(
        Sample=result
    )
end
