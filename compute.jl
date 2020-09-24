
filename = ARGS[1]

if isfile(filename * "_me.txt") || isfile(filename * "_ce.txt")
    error("File exists")
end

mefile = open(filename * "-me.txt", "w")
cefile = open(filename * "-ce.txt", "w")

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
ce = Trajectories.conditional_entropy(sn, rn, 100_000)

writedlm(mefile, me, ',')
writedlm(cefile, ce, ',')

close(mefile)
close(cefile)
