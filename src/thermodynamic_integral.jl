using Statistics
using FastGaussQuadrature
using LinearAlgebra
using DataFrames
using StatsFuns
using HDF5
using Statistics

import Random

mutable struct MetropolisSampler{S,Sys}
    burn_in::Int
    subsample::Int
    current_energy::Float64
    state::S
    chain::Sys
end

Base.iterate(sampler::MetropolisSampler) = iterate(sampler, deepcopy(sampler.state))

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

function generate_mcmc_samples(initial::State, chain::SignalChain, burn_in::Int, num_samples::Int) where State
    sampler = MetropolisSampler(burn_in, 0, energy(initial, chain), initial, chain)

    samples = Vector{State}(undef, num_samples)
    acceptance = zeros(Int16, num_samples)
    for (index, was_accepted) in Iterators.enumerate(Iterators.take(sampler, num_samples))
        samples[index] = deepcopy(sampler.state)
        acceptance[index] = was_accepted
    end

    samples, acceptance
end

function annealed_importance_sampling(initial, chain::SignalChain, subsample::Int, num_temps::Int)
    weights = zeros(Float64, num_temps)
    temps = range(0, 1; length=num_temps + 1)
    acceptance = zeros(Float64, num_temps)
    acceptance[1] = 1.0

    e_prev = energy(initial, chain.system, temps[1])
    e_cur = energy(initial, chain.system, temps[2])
    weights[1] = e_prev - e_cur

    chain.θ = temps[2]
    sampler = MetropolisSampler(0, subsample, e_cur, initial, chain)
    for (i, acc) in zip(2:num_temps, sampler)
        e_prev = sampler.current_energy
        e_cur = energy(sampler.state, chain.system, temps[i + 1])

        weights[i] = weights[i - 1] + e_prev - e_cur
        acceptance[i] = acc / (subsample + 1)

        # change the temperature for the next iteration
        sampler.chain.θ = temps[i + 1]
        sampler.current_energy = e_cur
    end
    temps, weights, acceptance
end

struct AnnealingEstimate
    subsample::Int
    num_temps::Int
    num_samples::Int
end

name(x::AnnealingEstimate) = "AIS"

abstract type SimulationResult end

struct AnnealingEstimationResult <: SimulationResult
    estimate::AnnealingEstimate
    inv_temps::Vector{Float64}
    weights::Array{Float64,2}
    acceptance::Array{Float64,2}
    initial_conditionals::Vector{Float64}
end

log_marginal(result::AnnealingEstimationResult) =  -(logsumexp(result.weights[end, :]) - log(size(result.weights, 2)))

function write_hdf5!(group, res_array::Vector{AnnealingEstimationResult})
    estimate = res_array[1].estimate
    attrs(group)["subsample"] = estimate.subsample
    attrs(group)["num_annealing_runs"] = estimate.num_samples
    attrs(group)["num_thetas"] = estimate.num_temps

    inv_temps = cat((r.inv_temps for r in res_array)...; dims=2)
    weights = cat((r.weights for r in res_array)...; dims=3)
    acceptance = cat((r.acceptance for r in res_array)...; dims=3)

    group["theta"] = inv_temps[:, 1]
    group["weights"] = weights[end, :, :]
    group["acceptance"] = mean(acceptance, dims=2)
    group["initial_conditional"] = hcat((r.initial_conditionals for r in res_array)...)

    weight_attrs = attrs(group["weights"])
    acceptance_attrs = attrs(group["acceptance"])

    weight_attrs["long_name"] = "Annealed Importance Sampling Weights"
    acceptance_attrs["long_name"] = "MCMC acceptance rates"

    weight_attrs["Coordinates"] = ["response_index", "annealing_run"]
    nothing
end

function simulate(algorithm::AnnealingEstimate, initial::Trajectory, system::StochasticSystem)
    chain = SignalChain(system, 1.0, 0.0, Float64[], Float64[])

    all_weights = zeros(Float64, algorithm.num_temps, algorithm.num_samples)
    acc = zeros(Float64, algorithm.num_temps, algorithm.num_samples)
    inv_temps = nothing
    initial_conditionals = zeros(algorithm.num_samples)
    for i in 1:algorithm.num_samples
        signal = new_signal(initial, system)
        initial_conditionals[i] = energy(signal, system, 1.0)
        (temps, weights, acceptance) = annealed_importance_sampling(signal, chain, algorithm.subsample, algorithm.num_temps)
        inv_temps = temps
        all_weights[:, i] = weights
        acc[:, i] = acceptance
    end

    AnnealingEstimationResult(algorithm, collect(inv_temps), all_weights, acc, initial_conditionals)
end

struct DirectMCEstimate
    num_samples::Int
end

name(x::DirectMCEstimate) = "Direct MC"

struct DirectMCResult
    samples::Vector{Float64}
end

function summary(res_array::AbstractVector{<:DirectMCResult})
    block_size = 2^14
    logmeanexp(x) = log(mean(exp.(x .- maximum(x)))) + maximum(x)
    blocks = [logmeanexp.(Iterators.partition(est.samples, block_size)) for est in res_array]
    DataFrame(Blocks=blocks)
end

log_marginal(result::DirectMCResult) = -(logsumexp(result.samples) - log(length(result.samples)))
function Statistics.var(result::DirectMCResult)
    max_weight = maximum(result.samples)
    log_mean_weight = max_weight + log(mean(exp.(result.samples .- max_weight)))
    log_var = log(var(exp.(result.samples .- max_weight))) + 2 * max_weight
    log_var -= log(length(result.samples))

    exp(-2log_mean_weight + log_var)
end


function simulate(algorithm::DirectMCEstimate, initial::Trajectory, system::StochasticSystem)
    samples = zeros(Float64, algorithm.num_samples)
    for i in 1:algorithm.num_samples
        signal = new_signal(initial, system)
        samples[i] = -energy(signal, system, 1.0)
    end
    DirectMCResult(samples)
end

struct TIEstimate
    burn_in::Int
    integration_nodes::Int
    num_samples::Int
end

name(x::TIEstimate) = "TI"

struct ThermodynamicIntegrationResult <: SimulationResult
    integration_weights::Vector{Float64}
    inv_temps::Vector{Float64}
    energies::Array{Float64,2}
    acceptance::Array{Bool,2}
end

function summary(res_array::AbstractVector{<:ThermodynamicIntegrationResult})
    block_size = 2^10
    energy_blocks = [blocks(r, block_size) for r in res_array]
    acceptance = [mean(r.acceptance, dims=1) for r in res_array]
    inv_temps = [r.inv_temps for r in res_array]
    integration_weights = [r.integration_weights for r in res_array]

    DataFrame(EnergyBlocks=energy_blocks, Acceptance=acceptance, Theta=inv_temps, IntegrationWeights=integration_weights)
end

function write_hdf5!(group, res_array::Vector{ThermodynamicIntegrationResult})
    inv_temps = cat((r.inv_temps for r in res_array)...; dims=2)
    integration_weights = cat((r.integration_weights for r in res_array)...; dims=2)
    # energies = cat((r.energies for r in res_array)...; dims=3)
    acceptance = cat((r.acceptance for r in res_array)...; dims=3)

    block_size = 2^10

    group["inv_temps"] = inv_temps[:, 1]
    group["integration_weights"] = integration_weights
    group["energy_blocks"] = cat((blocks(r, block_size) for r in res_array)..., dims=3)
    group["acceptance"] = mean(acceptance, dims=1)

    attrs(group["energy_blocks"])["block_size"] = block_size

    nothing
end

function write_hdf5!(group, dict::AbstractDict)
    for (name, value) in dict
        name = String(name)
        write_value_hdf5!(group, String(name), value)
    end
end

function write_hdf5!(group, df::AbstractDataFrame)
    for (name, value) in zip(names(df), eachcol(df))
        write_value_hdf5!(group, String(name), value)
    end
end

function write_value_hdf5!(group, name::String, value)
    # if we can't write the value directly as a dataset we fall back
    # to creating a new group
    newgroup = g_create(group, name)
    write_hdf5!(newgroup, value)
end

# non array values are written as arguments
function write_value_hdf5!(group, name::String, value::Union{String,Number})
    attrs(group)[name] = value
end

function write_value_hdf5!(group, name::String, value::AbstractArray{<:Number})
    group[name] = value
end

function write_value_hdf5!(group, name::String, value::AbstractVector{<:Array{T,N}}) where {T,N}
    outer_len = length(value)
    if outer_len < 1
        group[name] = zeros(T, 0)
        return
    end
    inner_size = size(value[1])
    dset = create_dataset(group, name, datatype(T), dataspace(inner_size..., outer_len), chunk=(inner_size..., 1))
    for (i, subarray) in enumerate(value)
        dset[axes(subarray)..., i] = subarray
    end
end

# perform the quadrature integral
log_marginal(result::ThermodynamicIntegrationResult) = dot(result.integration_weights, vec(mean(result.energies, dims=1)))
function Statistics.var(result::ThermodynamicIntegrationResult, block_size=2^10)
    b = blocks(result, block_size)
    σ² = var(b, dims=1) ./ size(b, 1)
    dot(result.integration_weights.^2, σ²)
end

function Statistics.var(result::AnnealingEstimationResult)
    max_weight = maximum(result.weights[end, :])
    log_mean_weight = max_weight + log(mean(exp.(result.weights[end,:] .- max_weight)))
    log_var = log(var(exp.(result.weights[end, :] .- max_weight))) + 2 * max_weight
    log_var -= log(size(result.weights, 2))

    exp(-2log_mean_weight + log_var)
end

function blocks(result::ThermodynamicIntegrationResult, block_size=2^10)
    block_averages(array) = map(mean, Iterators.partition(array, block_size))
    mapreduce(block_averages, hcat, eachcol(result.energies))
end

# Monte-Carlo computation of the marginal probability for the given configuration
function simulate(algorithm::TIEstimate, initial::Trajectory, system::StochasticSystem)
    # Generate the array of θ values for which we want to simulate the system.
    # We use Gauss-Legendre quadrature which predetermines the choice of θ.
    nodes, weights = gausslegendre(algorithm.integration_nodes)
    θrange = 0.5 .* nodes .+ 0.5
    # The factor 0.5 comes from rescaling the integration limits from [-1,1] to [0,1].
    weights = 0.5 .* weights

    energies = zeros(Float64, algorithm.num_samples, length(θrange))
    accept = Array{Bool}(undef, algorithm.num_samples, length(θrange))
    for i in eachindex(θrange)
        chn = chain(system, θrange[i])
        sampler = MetropolisSampler(algorithm.burn_in, 0, energy(initial, chn), deepcopy(initial), chn)
        for (j, was_accepted) in Iterators.enumerate(Iterators.take(sampler, algorithm.num_samples))
            energies[j, i] = energy(sampler.state, system, 1.0)
            accept[j, i] = was_accepted != 0
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
    stats = DataFrame(
        Sample=zeros(Float64, num_responses), 
        Variance=zeros(Float64, num_responses), 
        TimeElapsed=zeros(Float64, num_responses), 
        GcTime=zeros(Float64, num_responses),
        InitialEnergy=zeros(Float64, num_responses)
    )

    results = map(1:num_responses) do i
        (system, initial) = generate_configuration(gen; duration=duration)
        stats.InitialEnergy[i] = energy(initial, system, 1.0)

        timed_result = @timed simulate(algorithm, initial, system)

        sample = log_marginal(timed_result.value)
        variance = var(timed_result.value)

        stats.Sample[i] = sample
        stats.Variance[i] = variance
        stats.TimeElapsed[i] = timed_result.time
        stats.GcTime[i] = timed_result.gctime

        timed_result.value
    end

    Dict("marginal_entropy" => stats, "marginal_entropy_estimate" => summary(results))
end


function conditional_entropy(gen::ConfigurationGenerator;  num_responses::Int=1, duration::Float64=500.0)
    result = zeros(Float64, num_responses)
    for i in 1:num_responses
        (system, initial) = generate_configuration(gen, duration=duration)
        result[i] = energy(initial, system, 1.0)
    end

    Dict("conditional_entropy" => DataFrame(
        Sample=result
    ))
end
