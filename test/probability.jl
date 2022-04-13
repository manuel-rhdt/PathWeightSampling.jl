using Catalyst
using PathWeightSampling
using Test
using StaticArrays
using Distributions

sn = @reaction_network begin
    κ, ∅ --> S
    λ, S --> ∅
end κ λ

κ = 0.25
λ = 0.005

dist = PathWeightSampling.distribution(sn, [κ, λ])

traj = PathWeightSampling.Trajectory([[50.0], [51.0], [50.0]], [1.0, 2.0, 3.0], [1, 2])

p_wait(s, dt) = exp(- (κ + s * λ) * dt)

wait1 = log(p_wait(50, 1.0))
wait2 = log(p_wait(51, 1.0))
wait3 = log(p_wait(50, 1.0))
reac1 = log(κ)
reac2 = log(51 * λ)

@test PathWeightSampling.logpdf(dist, traj) ≈ sum((wait1, wait2, wait3, reac1, reac2))
@test PathWeightSampling.trajectory_energy(dist, traj) ≈ sum((wait1, wait2, wait3, reac1, reac2))
@test PathWeightSampling.trajectory_energy(dist, traj, tspan=(1.0,2.0)) ≈ sum((wait2, reac2))

cumulative = PathWeightSampling.cumulative_logpdf(dist, traj, 0:0.5:3)
@test cumulative ≈ [
    0.0,
    0.5wait1,
    wait1 + reac1,
    wait1 + reac1 + 0.5wait2,
    wait1 + reac1 + wait2 + reac2,
    wait1 + reac1 + wait2 + reac2 + 0.5wait3,
    wait1 + reac1 + wait2 + reac2 + wait3,
]

subset = PathWeightSampling.cumulative_logpdf(dist, traj, 2:0.33:3)
@test subset ≈ [
    0.0,
    0.33wait3,
    0.66wait3,
    0.99wait3
]

traj2 = PathWeightSampling.Trajectory([[50.0]], [1.0])
@test PathWeightSampling.trajectory_energy(dist, traj2, tspan=(0.0,0.5)) == -0.25

