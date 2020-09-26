
dir_name = ARGS[1]

path = mkdir(dir_name)
me_fname = joinpath(path, "me.txt")
ce_fname = joinpath(path, "ce.txt")

using Logging

mefile = open(me_fname, "w")
@info "Created File" me_fname
cefile = open(ce_fname, "w")
@info "Created File" ce_fname

using Distributed
addprocs(exeflags="--project")

@everywhere using GaussianMcmc.Trajectories
using Catalyst
using CSV

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

gen = Trajectories.configuration_generator(sn, rn)

me = @distributed (vcat) for i = 1:8
    Trajectories.marginal_entropy(gen; num_responses=1, num_samples=2000, integration_nodes=16, duration=100.0)
end
CSV.write(mefile, me)
close(mefile)
@info "Finished marginal entropy"

ce = Trajectories.conditional_entropy(gen, num_responses=10_000, duration=500.0)
CSV.write(cefile, ce)
close(cefile)
@info "Finished conditional entropy"
