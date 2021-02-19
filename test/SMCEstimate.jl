import GaussianMcmc: chemotaxis_system, SMCEstimate, DirectMCEstimate, mutual_information

system = chemotaxis_system()
smc_est = SMCEstimate(1_000)
result_smc = mutual_information(system, smc_est, num_responses = 2)
