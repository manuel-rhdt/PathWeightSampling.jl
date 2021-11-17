
import PathWeightSampling
using Test

system = PathWeightSampling.gene_expression_system(dtimes=0:0.2:1.0)
conf = PathWeightSampling.generate_configuration(system)
ensemble = PathWeightSampling.MarginalEnsemble(system)

# Test the shooting moves
jump_problem = ensemble.jump_problem
original_traj = conf.s_traj
new_traj = copy(conf.s_traj)
@test original_traj == new_traj
PathWeightSampling.shoot_forward!(new_traj, original_traj, jump_problem, 0.5)
@test original_traj != new_traj
@test issorted(new_traj.t) && allunique(new_traj.t)
@test length(new_traj.t) == length(new_traj.u) == length(new_traj.i)

new_traj = copy(conf.s_traj)
@test original_traj == new_traj
PathWeightSampling.shoot_backward!(new_traj, original_traj, jump_problem, 0.5)
@test original_traj != new_traj
@test issorted(new_traj.t) && allunique(new_traj.t)
@test length(new_traj.t) == length(new_traj.u) == length(new_traj.i)

mchain = PathWeightSampling.chain(ensemble; θ=1.0)
@test PathWeightSampling.energy(conf, mchain) < 0


# Test Thermodynamic Integration

alg = PathWeightSampling.TIEstimate(100, 16, 1000)
result = PathWeightSampling.simulate(alg, conf, ensemble)
@test all(PathWeightSampling.log_marginal(result) .>= 0)
