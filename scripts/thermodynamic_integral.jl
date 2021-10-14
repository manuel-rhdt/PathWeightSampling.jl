include("basic_setup.jl")

algorithm = TIEstimate(1024, 16, 2^16)

using Statistics
using Plots
using LaTeXStrings


durations = range(0.2, 2.0, length=4)
results = Vector{PWS.ThermodynamicIntegrationResult}(undef, length(durations))
log_likelihoods = Vector{Float64}(undef, length(durations))

Threads.@threads for i ∈ eachindex(durations)
    system = get_system(20.0, 20.0, 1.0, 0.1, durations[i])
    initial = PWS.generate_configuration(system)
    result = Base.invokelatest(PWS.simulate, algorithm, initial, system)
    results[i] = result
    log_likelihoods[i] = -Base.invokelatest(PWS.energy, initial, system, 1.0)
end

p = plot()
for (r, duration) in zip(results, durations)
    plot!(p, r.inv_temps, mean(r.acceptance, dims=1)', ylim=(0, 1), label=L"T=%$duration", ylabel="acceptance rate", xlabel=L"\theta", legend=:bottomleft)
end
savefig(p, projectdir("plots", "acceptance_rates.png"))
p

p = plot()
for (r, duration, ll) in zip(results, durations, log_likelihoods)
    y = -vec(mean(r.energies, dims=1)) .+ minimum(mean(r.energies, dims=1))
    yerr = vec(sqrt.(var(r.energies, dims=1) / (size(r.energies, 1) / 25)))
    plot!(p, r.inv_temps, y, yerr=yerr, label=L"T=%$duration", xlabel=L"\theta", ylabel=L"E_\theta[\ln\mathrm{P}(x|s)] - \ln\mathrm{P}(x|s_0)", legend=:bottomright)
end
p
savefig(p, projectdir("plots", "thermodynamic_integral.png"))

