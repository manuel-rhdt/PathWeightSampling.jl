using PathWeightSampling
using Test
using DiffEqJump
using StaticArrays

system = PathWeightSampling.cooperative_chemotaxis_system(aggregator = Direct())

seed = 1234
conf = PathWeightSampling.generate_configuration(system; seed = seed)

@test conf.s_traj.t[end] == system.dtimes[end]
@test conf.x_traj.t[end] == system.dtimes[end]

full_conf = PathWeightSampling.generate_full_configuration(system; seed = seed)

@test conf.s_traj == full_conf.s_traj
@test conf.x_traj == full_conf.x_traj
