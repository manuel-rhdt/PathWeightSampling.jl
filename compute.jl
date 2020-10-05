
dir_name = ARGS[1]

path = mkdir(dir_name)
me_fname = joinpath(path, "me.h5")
ce_fname = joinpath(path, "ce.txt")

using Logging


using Distributed
addprocs(exeflags="--project")

@everywhere using GaussianMcmc.Trajectories
using Catalyst
using CSV
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

algorithm = AnnealingEstimate(10, 50, 1000)

# using distributed because of GarbageCollector pauses in Multi-Threaded code
me = @distributed (vcat) for i = 1:1
    result, stats = Trajectories.marginal_entropy(gen, algorithm=algorithm; num_responses=1, duration=500.0)
    println(stats)
    result
end

h5open(me_fname, "w") do file
    Trajectories.write_hdf5!(file, me)
end
@info "Finished marginal entropy"

cefile = open(ce_fname, "w")
ce = Trajectories.conditional_entropy(gen, num_responses=10_000, duration=500.0)
CSV.write(cefile, ce)
close(cefile)
@info "Finished conditional entropy"
