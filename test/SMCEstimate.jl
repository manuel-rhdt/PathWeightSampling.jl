import GaussianMcmc: chemotaxis_system, SMCEstimate, mutual_information

system = chemotaxis_system()
algorithm = SMCEstimate(1_000)
result = mutual_information(system, algorithm, num_responses = 5)

last.(result[!, :MutualInformation])