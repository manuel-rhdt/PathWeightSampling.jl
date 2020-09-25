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
(system, s1) = Trajectories.generate_configuration(gen)
s2 = deepcopy(s1)
scopy = deepcopy(s1)


Trajectories.shoot_forward!(s2, s1, system.jump_problem)
@test issorted(s2.t)
@test s1.t != s2.t
@test s1.u != s2.u
@test s1.u[1] == s2.u[1]
@test scopy == s1

Trajectories.shoot_backward!(s2, s1, system.jump_problem)
@test issorted(s2.t)
@test s1.t != s2.t
@test s1.u != s2.u
@test s1.u[end] == s2.u[end]
@test scopy == s1
