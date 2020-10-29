using DrWatson
import JSON

f = ARGS[1]
dict = JSON.parsefile(projectdir("_research", "tmp", f))
@info "Read file" file = projectdir("_research", "tmp", f)

duration = dict["duration"]
N = dict["N"]
num_responses = dict["num_responses"]
run_name = dict["run_name"]

using GaussianMcmc.Trajectories
using HDF5
using Logging
using Catalyst

if dict["algorithm"] == "thermodynamic_integration"
    algorithm = TIEstimate(1024, 6, 2^14)
elseif dict["algorithm"] == "annealing"
    algorithm = AnnealingEstimate(5, 100, 100)
else
    error("Unsupported algorithm " * dict["algorithm"])
end

@info "Parameters" run_name duration N num_responses algorithm

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
@info "Finished marginal entropy"
conditional_entropy = Trajectories.conditional_entropy(gen, num_responses=10_000, duration=duration)
@info "Finished conditional entropy"

function DrWatson._wsave(filename, result::Dict)
    h5open(filename, "w") do file
        Trajectories.write_hdf5!(file, result)
    end
end


filename = savename((@dict duration N num_responses), "hdf5")
local_path = datadir(dict["algorithm"], run_name, filename)
tagsave(local_path, merge(dict, marginal_entropy, conditional_entropy))
@info "Saved to" filename

# upload to SUN storage

include("smbclient.jl")
sun_path = joinpath(sun_home, "data", dict["algorithm"], run_name, filename)
mkpath(dirname(sun_path))
cp(local_path, sun_path)
