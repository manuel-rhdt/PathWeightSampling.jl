using Catalyst
using GaussianMcmc.Trajectories

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

dist = GaussianMcmc.Trajectories.distribution(sn)

for i in 1:100
    @test dist.totalrate([i], []) ≈ 0.005 * i + 0.25
end