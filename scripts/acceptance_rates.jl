include("basic_setup.jl")

t_s = 100
t_x = 1
κ = mean_s / t_s
λ = 1 / t_s
ρ = 1 / t_x
μ = 1 / t_x
mean_x = mean_s * ρ / μ
gen = Trajectories.configuration_generator(sn, rn, [κ, λ], [ρ, μ], mean_s, mean_x)
system, initial = Trajectories.generate_configuration(gen, duration=100.0)

chain = Trajectories.chain(system, 1.0)
samples, acceptance = Trajectories.generate_mcmc_samples(initial, chain, 2^8, 2^18)

using Plots
using Statistics
using StatsBase
using LaTeXStrings

hist_acc = fit(Histogram, chain.accepted_list, nbins=30)
hist_rej = fit(Histogram, chain.rejected_list, hist_acc.edges[1])

acc_rate = hist_acc.weights ./ (hist_rej.weights .+ hist_acc.weights)

gr()
p = plot(
    hist_acc.edges[1], 
    acc_rate, 
    seriestype=:barbins,
    ylim=(0,1),
    legend=false,
    xlabel="duration of regrown segment",
    ylabel="ratio of accepted MCMC moves",
    size=(270,270),
)

savefig(p, projectdir("plots", "ratio_acceptance_rates.png"))
savefig(p, projectdir("plots", "ratio_acceptance_rates.tex"))

