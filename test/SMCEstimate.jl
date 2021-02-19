import GaussianMcmc: chemotaxis_system, SMCEstimate, DirectMCEstimate, mutual_information

system = chemotaxis_system()
smc_est = SMCEstimate(1_000)
dmc_est = DirectMCEstimate(1_000)
result_smc = mutual_information(system, smc_est, num_responses = 20)
result_dmc = mutual_information(system, dmc_est, num_responses = 20)

using Plots

p = plot(system.dtimes, result_smc[:, :MutualInformation], color=1, label="")
plot!(p, system.dtimes, result_dmc[:, :MutualInformation], color=2, label="")
plot!(p, 1, label="Sequential MC", color=1)
plot!(p, 1, label="Brute Force", color=2, legend=:bottomleft, xlabel="duration", ylabel="Path Mutual Information", title="Estimation Variance Comparison", dpi=300)

savefig(p, "~/Downloads/estimate_variance.png")