
dir_name = ARGS[1]

path = mkdir(dir_name)
result_name = joinpath(path, "sim.h5")

using Logging

using GaussianMcmc.Trajectories
using Catalyst
using HDF5

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

gen = Trajectories.configuration_generator(sn, rn)

# algorithm = AnnealingEstimate(10, 50, 100)
algorithm = TIEstimate(1024, 16, 2^14)

marginal_entropy = Trajectories.marginal_entropy(gen, algorithm=algorithm; num_responses=5, duration=100.0)
conditional_entropy = Trajectories.conditional_entropy(gen, num_responses=10_000, duration=500.0)

result = merge(marginal_entropy, conditional_entropy)
h5open(result_name, "w") do file
    Trajectories.write_hdf5!(file, result)
end
