# Copyright 2021 Manuel Reinhardt
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import GaussianMcmc
using Test

system = GaussianMcmc.gene_expression_system(dtimes=0:0.2:1.0)
conf = GaussianMcmc.generate_configuration(system)
ensemble = GaussianMcmc.MarginalEnsemble(system)

# Test the shooting moves
jump_problem = ensemble.jump_problem
original_traj = conf.s_traj
new_traj = copy(conf.s_traj)
@test original_traj == new_traj
GaussianMcmc.shoot_forward!(new_traj, original_traj, jump_problem, 0.5)
@test original_traj != new_traj
@test issorted(new_traj.t) && allunique(new_traj.t)
@test length(new_traj.t) == length(new_traj.u) == length(new_traj.i)

new_traj = copy(conf.s_traj)
@test original_traj == new_traj
GaussianMcmc.shoot_backward!(new_traj, original_traj, jump_problem, 0.5)
@test original_traj != new_traj
@test issorted(new_traj.t) && allunique(new_traj.t)
@test length(new_traj.t) == length(new_traj.u) == length(new_traj.i)

mchain = GaussianMcmc.chain(ensemble; θ=1.0)
@test GaussianMcmc.energy(conf, mchain) < 0


# Test Thermodynamic Integration

alg = GaussianMcmc.TIEstimate(100, 16, 1000)
result = GaussianMcmc.simulate(alg, conf, ensemble)
@test all(GaussianMcmc.log_marginal(result) .>= 0)

GaussianMcmc.mutual_information(system, alg)
