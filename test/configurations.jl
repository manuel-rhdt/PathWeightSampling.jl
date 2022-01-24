using PathWeightSampling
using Test
using DiffEqJump
using StaticArrays
using Catalyst
using ModelingToolkit

system = PathWeightSampling.cooperative_chemotaxis_system(aggregator = Direct(), mmax = 9)

joint = PathWeightSampling.reaction_network(system)

@test ModelingToolkit.tosymbol.(Catalyst.species(joint)) ==
      ModelingToolkit.tosymbol.(vcat((PathWeightSampling.independent_species(x) for x in [system.sn, system.rn, system.xn])...))


seed = 1234
conf = PathWeightSampling.generate_configuration(system; seed = seed)

@test conf.s_traj.t[end] == system.dtimes[end]
@test conf.x_traj.t[end] == system.dtimes[end]

full_conf = PathWeightSampling.generate_full_configuration(system; seed = seed)

@test conf.s_traj == full_conf.s_traj
@test conf.x_traj == full_conf.x_traj

cens = PathWeightSampling.ConditionalEnsemble(system)
full_conf2 = PathWeightSampling.sample(conf, cens)
@test full_conf.s_traj == full_conf2.s_traj
@test full_conf.r_traj != full_conf2.r_traj
@test full_conf.x_traj == full_conf2.x_traj


