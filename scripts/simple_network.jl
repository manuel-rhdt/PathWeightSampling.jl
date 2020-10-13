using DrWatson

f = ARGS[1]
dict = load(projectdir("_research", "tmp", f))

duration = dict["duration"]
N = dict["N"]
num_responses = dict["num_responses"]

using GaussianMcmc.Trajectories
using HDF5
using Logging
using Catalyst

if dict["algorithm"] == "thermodynamic_integration"
    algorithm = TIEstimate(1024, 16, 2^14)
elseif dict["algorithm"] == "annealing"
    algorithm = AnnealingEstimate(10, 50, 100)
else
    error("Unsupported algorithm " * dict["algorithm"])
end

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

gen = Trajectories.configuration_generator(sn, rn)
marginal_entropy = Trajectories.marginal_entropy(gen, algorithm=algorithm; num_responses=num_responses, duration=duration)
conditional_entropy = Trajectories.conditional_entropy(gen, num_responses=10_000, duration=duration)


function DrWatson._wsave(filename, result::Dict)
    h5open(filename, "w") do file
        Trajectories.write_hdf5!(file, result)
    end
end


filename = savename((@dict duration N), "hdf5")
tagsave(datadir(dict["algorithm"], filename), merge(dict, marginal_entropy, conditional_entropy))
