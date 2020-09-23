
filename = ARGS[1]

if isfile(filename * "_me.txt") || isfile(filename * "_ce.txt")
    error("File exists")
end

using GaussianMcmc.Trajectories
using Catalyst

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

me = Trajectories.marginal_entropy(sn, rn, 2000, 2000, 16)
ce = Trajectories.conditional_entropy(sn, rn, 100_000)

using DelimitedFiles

open(filename * "_me.txt", "w") do io
    writedlm(io, me, ',')
end

open(filename * "_ce.txt", "w") do io
    writedlm(io, ce, ',')
end
