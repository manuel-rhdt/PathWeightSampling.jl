using GaussianMcmc.Trajectories
using Catalyst
using StaticArrays
using DifferentialEquations
using Test

sn = @reaction_network begin
    0.005, S --> ∅
    0.25, ∅ --> S
end

rn = @reaction_network begin
    0.01, S --> X + S
    0.01, X --> ∅ 
end

gen = Trajectories.configuration_generator(sn, rn)
conf1 = Trajectories.generate_configuration(gen)
conf2 = copy(conf1)

Trajectories.shoot_forward!(conf2.signal, conf1.signal, conf1.jump_problem)

@test issorted(conf2.signal.t)
@test conf1.signal.u[1] == conf2.signal.u[1]

Trajectories.shoot_backward!(conf2.signal, conf1.signal, conf1.jump_problem)
@test conf1.signal.u[end] == conf2.signal.u[end]
@test issorted(conf2.signal.t)
