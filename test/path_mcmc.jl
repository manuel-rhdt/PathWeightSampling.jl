import PathWeightSampling as PWS
using Test

rates = [1.0, 1.0, 1.0, 1.0]
rstoich = [[], [1 => 1], [1 => 1], [2 => 1]]
nstoich = [[1 => 1], [1 => -1], [2 => 1], [2 => -1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:S, :X])

N = 10
state_range = CartesianIndices((0:N, 0:N))
linear_range = LinearIndices(state_range)
state_to_index(state) = linear_range[CartesianIndex(state) - state_range[begin] + CartesianIndex(1, 1)]

p_init = zeros(N + 1, N + 1)
p_init[state_to_index((5, 5))] = 1.0
p_init

state_sequence = PWS.MarkovMcmc.propagate_probs(p_init, 0:0.1:1.0, reactions, N, 100.0)

@test all(length.(state_sequence) .== 121)
@test all(sum.(state_sequence) .≈ 1.0)

sum(state_sequence[2])
state_sequence[10]