include("basic_setup.jl")

algorithm = TIEstimate(1024, 16, 2^16)


using Statistics
using Plots
using LaTeXStrings


durations = range(50.0, 500, length=4)
results = Vector{Trajectories.ThermodynamicIntegrationResult}(undef, length(durations))
log_likelihoods = Vector{Float64}(undef, length(durations))

Threads.@threads for i ∈ eachindex(durations)
    (system, initial) = Trajectories.generate_configuration(gen; duration=durations[i])
    result = Trajectories.simulate(algorithm, initial, system)
    results[i] = result
    log_likelihoods[i] = -Trajectories.energy(initial, system, θ=1.0)
end

p = plot(dpi=300)
for (r, duration) in zip(results, durations)
    plot!(p, r.inv_temps, mean(r.acceptance, dims=1)', ylim=(0,1), label=L"T=%$duration", ylabel="acceptance rate", xlabel=L"\theta", legend=:bottomleft)
end
savefig(p, projectdir("plots", "acceptance_rates.png"))

p = plot(dpi=300)
for (r, duration, ll) in zip(results, durations, log_likelihoods)
    y = -vec(mean(r.energies, dims=1)) .- ll
    yerr = vec(sqrt.(var(r.energies, dims=1)/(size(r.energies, 1) / 25)))
    plot!(p, r.inv_temps, y, yerr=yerr, label=L"T=%$duration", xlabel=L"\theta", ylabel=L"E_\theta[\ln\mathrm{P}(x|s)] - \ln\mathrm{P}(x|s_0)", legend=:bottomright)
end
savefig(p, projectdir("plots", "thermodynamic_integral.png"))

