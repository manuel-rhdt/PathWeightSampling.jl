import StatsFuns: logsumexp

struct AnnealingEstimate
    subsample::Int
    num_temps::Int
    num_samples::Int
end

name(x::AnnealingEstimate) = "AIS"

function annealed_importance_sampling(initial, chain::MarkovChain, subsample::Int, num_temps::Int)
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

struct AnnealingEstimationResult <: SimulationResult
    estimate::AnnealingEstimate
    inv_temps::Vector{Float64}
    weights::Array{Float64,2}
    acceptance::Array{Float64,2}
    initial_conditionals::Vector{Float64}
end

log_marginal(result::AnnealingEstimationResult) =  -(logsumexp(result.weights[end, :]) - log(size(result.weights, 2)))

function simulate(algorithm::AnnealingEstimate, initial, system)
    chain = SignalChain(system, 1.0, 0.0, Float64[], Float64[])

    all_weights = zeros(Float64, algorithm.num_temps, algorithm.num_samples)
    acc = zeros(Float64, algorithm.num_temps, algorithm.num_samples)
    inv_temps = nothing
    initial_conditionals = zeros(algorithm.num_samples)
    for i in 1:algorithm.num_samples
        initial_energy = Inf
        signal = initial
        while isinf(initial_energy)
            signal = new_signal(initial, system)
            initial_energy = energy(signal, system, 1.0)
        end

        initial_conditionals[i] = initial_energy
        (temps, weights, acceptance) = annealed_importance_sampling(signal, chain, algorithm.subsample, algorithm.num_temps)
        inv_temps = temps
        all_weights[:, i] = weights
        acc[:, i] = acceptance
    end

    AnnealingEstimationResult(algorithm, collect(inv_temps), all_weights, acc, initial_conditionals)
end

function Statistics.var(result::AnnealingEstimationResult)
    max_weight = maximum(result.weights[end, :])
    log_mean_weight = max_weight + log(mean(exp.(result.weights[end,:] .- max_weight)))
    log_var = log(var(exp.(result.weights[end, :] .- max_weight))) + 2 * max_weight
    log_var -= log(size(result.weights, 2))

    exp(-2log_mean_weight + log_var)
end

function summary(results::AnnealingEstimationResult...)
    DataFrame(Weights=[r.weights[end, :] for r in results])
end