using StaticArrays
using Test

include("test_system.jl")

(system, s1) = Trajectories.generate_configuration(gen, duration=20.0)
s2 = deepcopy(s1)
scopy = deepcopy(s1)

branch_time = 10.0

Trajectories.shoot_forward!(s2, s1, system.jump_problem, branch_time)
@test issorted(s2.t)
@test s1.t != s2.t
@test s1.u != s2.u
@test s1(10.0) == s2(10.0)
@test scopy == s1

Trajectories.shoot_backward!(s2, s1, system.jump_problem, branch_time)
@test issorted(s2.t)
@test all(diff(s2.t) .!= 0.0)
@test s1.t != s2.t
@test s1.u != s2.u
@test s1(10.0) == s2(10.0)
@test scopy == s1
