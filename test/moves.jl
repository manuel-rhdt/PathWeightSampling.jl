using StaticArrays
using Test

include("test_system.jl")

s1 = GaussianMcmc.generate_configuration(system).signal
s2 = deepcopy(s1)
scopy = deepcopy(s1)

branch_time = 0.5

GaussianMcmc.shoot_forward!(s2, s1, system.signal_j_problem, branch_time)
@test issorted(s2.t)
@test s1.t != s2.t
@test s1.u != s2.u
@test s1(0.5) == s2(0.5)
@test scopy == s1

GaussianMcmc.shoot_backward!(s2, s1, system.signal_j_problem, branch_time)
@test issorted(s2.t)
@test all(diff(s2.t) .!= 0.0)
@test s1.t != s2.t
@test s1.u != s2.u
@test s1(0.5) == s2(0.5)
@test scopy == s1
