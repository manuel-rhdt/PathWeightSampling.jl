include("basic_setup.jl")
using Plots
using Statistics
using StatsBase

system.θ = 1.0
block(arr) = 0.5 .* (arr[begin:2:end-1] .+ arr[begin+1:2:end])

function plot_block_averages!(p, values)
    a = values

    y_vals = Float64[]
    y_err = (Float64[], Float64[])

    while length(a) > 1
        normed_variance = var(a, corrected=false) / (length(a) - 1)
        variance_err = sqrt(2*normed_variance^2 / (length(a) - 1))

        push!(y_vals, sqrt(normed_variance))
        push!(y_err[1], -sqrt(max(normed_variance - variance_err, 0)) + sqrt(normed_variance))
        push!(y_err[2], sqrt(normed_variance + variance_err) - sqrt(normed_variance))

        a = block(a)
    end

    plot!(p, (1:length(y_vals)), y_vals, yerror=y_err)
end

p = plot()
for num_samples ∈ [14, 16, 18]
    samples, acceptance = Trajectories.generate_mcmc_samples(signal, system, 2^10, 2^num_samples)
    energies = Trajectories.energy.(samples, Ref(system))
    plot_block_averages!(p, energies)
end

p
