import GaussianMcmc: chemotaxis_system, SMCEstimate, DirectMCEstimate, mutual_information

system = chemotaxis_system()
smc_est = SMCEstimate(1_000)
dmc_est = DirectMCEstimate(1_000)
result_smc = mutual_information(system, smc_est, num_responses = 5)
result_dmc = mutual_information(system, dmc_est, num_responses = 5)

result_smc[1, :MutualInformation]
result_dmc[1, :MutualInformation]