include("basic_setup.jl")

function get_results(algorithm, num_evals)
    results = zeros(num_evals)
    Threads.@threads for i in 1:num_evals
        r = Trajectories.simulate(algorithm, initial, system)
        results[i] = Trajectories.log_marginal(r)
    end
    results
end

an = AnnealingEstimate(15, 50, 100)
an_samples = get_results(an, 64)

ti = TIEstimate(1024, 6, 2^13)
ti_samples = get_results(ti, 64)

direct = DirectMCEstimate(50_000)
di_samples = get_results(direct, 64)

using Plots
using StatsPlots
using Statistics

jitter_ti = randn(size(ti_samples)...)
jitter_an = randn(size(an_samples)...)
jitter_di = randn(size(di_samples)...)
p = scatter(jitter_ti, ti_samples, label="TI", xlim=(-15, 15))
scatter!(jitter_an, an_samples, label="AIS")
scatter!(jitter_di, di_samples, label="Direct MC")
plot!(p, [-3,3], mean(ti_samples) .* ones(2))
plot!(p, [-3,3], mean(an_samples) .* ones(2))
plot!(p, [-3,3], mean(di_samples) .* ones(2))

density([ti_samples, an_samples, di_samples], label=hcat("TI", "AIS", "Direct MC"))