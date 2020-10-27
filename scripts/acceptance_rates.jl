include("basic_setup.jl")

Trajectories.reset(system)
samples, acceptance = Trajectories.generate_mcmc_samples(initial, system, 2^8, 2^18)

using Plots
using Statistics
using StatsBase
using LaTeXStrings

hist_acc = fit(Histogram, system.accepted_list, nbins=30)
hist_rej = fit(Histogram, system.rejected_list, hist_acc.edges[1])

acc_rate = hist_acc.weights ./ (hist_rej.weights .+ hist_acc.weights)

pgfplotsx()
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

