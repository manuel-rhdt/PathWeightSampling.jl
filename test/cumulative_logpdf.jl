import PathWeightSampling: DirectMCEstimate, chemotaxis_system, ConditionalEnsemble, MarginalEnsemble, marginal_configuration, generate_configuration, collect_samples, mutual_information
using Test
using Statistics

system = chemotaxis_system()

algorithm = DirectMCEstimate(1000)
dtimes = collect(range(0.0, 2.0, length=51)[2:end])
num_samples = 10

result = mutual_information(system, algorithm, num_samples=num_samples, dtimes=dtimes)

mi = hcat(result.MutualInformation...)

@test all(isfinite.(mi))

mean(mi, dims=2)
