
filename = ARGS[1]

me_fname = filename * "-me.txt"
ce_fname = filename * "-ce.txt"

if isfile(me_fname) || isfile(ce_fname)
    error("File exists")
end

using Logging

mefile = open(me_fname, "w")
@info "Created File" me_fname
cefile = open(ce_fname, "w")
@info "Created File" ce_fname

using GaussianMcmc.Trajectories
using Catalyst
using DelimitedFiles

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

me = Trajectories.marginal_entropy(sn, rn, 200, 2000, 16)

@info "Finished marginal entropy"

ce = Trajectories.conditional_entropy(sn, rn, 100_000)

@info "Finished conditional entropy"

writedlm(mefile, me, ',')
writedlm(cefile, ce, ',')

close(mefile)
close(cefile)
