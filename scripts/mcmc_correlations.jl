using GaussianMcmc.Trajectories
using Catalyst

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

gen = Trajectories.configuration_generator(sn, rn)

(system, initial) = Trajectories.generate_configuration(gen; duration=500.0)
signal = Trajectories.new_signal(initial, system)

system.θ = 1.0

samples, acceptance = Trajectories.generate_mcmc_samples(signal, system, 2^10, 2^16)

using Plots
using Statistics
using StatsBase

energies = Trajectories.energy.(samples, Ref(system))

plot(energies)
plot(autocor(energies))

mean(acceptance)

block(arr) = 0.5 .* (arr[begin:2:end-1] .+ arr[begin+1:2:end])

function plot_block_averages(values)
    a = values

    y_vals = Float64[]
    y_err = (Float64[], Float64[])

    while length(a) > 10
        normed_variance = var(a, corrected=false) / (length(a) - 1)
        variance_err = sqrt(2*normed_variance^2 / (length(a) - 1))

        push!(y_vals, sqrt(normed_variance))
        push!(y_err[1], -sqrt(max(normed_variance - variance_err, 0)) + sqrt(normed_variance))
        push!(y_err[2], sqrt(normed_variance + variance_err) - sqrt(normed_variance))

        a = block(a)
    end

    plot((1:length(y_vals)), y_vals, yerror=y_err)
end

plot_block_averages(energies)