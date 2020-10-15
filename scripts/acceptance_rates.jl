include("basic_setup.jl")

system
initial

reset(system)
samples, acceptance = Trajectories.generate_mcmc_samples(initial, system, 2^10, 2^18)

using Plots
using Statistics
using StatsBase
using LaTeXStrings

hist_acc = fit(Histogram, system.accepted_list, nbins=35)
hist_rej = fit(Histogram, system.rejected_list, hist_acc.edges[1], closed=:left)

acc_rate = hist_acc.weights ./ (hist_rej.weights + hist_acc.weights)

p = plot(
    hist_acc.edges[1], 
    acc_rate, 
    seriestype=:barbins,
    title="Ratio of accepted MCMC moves",
    legend=false,
    xlabel="Regrowth length",
    dpi=300,
    ylabel=L"\frac{N^{(\textrm{accepted})}}{N^{(\textrm{accepted})} + N^{(\textrm{rejected})}}"
)

savefig(p, projectdir("plots", "ratio_acceptance_rates.png"))