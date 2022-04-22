using PathWeightSampling
using DiffEqJump
using StochasticDiffEq
using Catalyst
using Test

@variables t L(t)
@parameters κ λ

D = Differential(t)

eqs = [D(L) ~ λ * (κ / λ - L)]
noiseeqs = [sqrt(κ / λ)]

@named sn = SDESystem(eqs, noiseeqs, t, [L], [κ, λ])

rn = @reaction_network rn begin
    ρ, L + R --> L + LR
    μ, LR --> R
end ρ μ

xn = @reaction_network xn begin
    δ, LR + Y --> Yp + LR
    χ, Yp --> Y
end δ χ

# create a SDE driven jump problem:

tspan = (0.0, 5.0)
u0s = [50.0]
ps = [50.0, 1.0]
sde_problem = SDEProblem(sn, u0s, tspan, ps)

u0r = [50.0, 100.0, 0.0]
pr = [50.0 / 50, 50.0]
discrete_problem = DiscreteProblem(rn, u0r, tspan, pr)
jump_problem = JumpProblem(rn, discrete_problem, Direct())

driven_jump_problem = PathWeightSampling.DrivenJumpProblem(jump_problem, sde_problem, save_jumps=true)
sol = solve(driven_jump_problem)

sds = PathWeightSampling.SDEDrivenSystem(
    sn, rn, xn,
    [50, 100, 0, 100, 0], # u0
    ps,
    pr,  # pr
    [1.0, 10.0],  # px
    0.0:0.1:5.0  # dtimes
)

conf = PathWeightSampling.generate_configuration(sds)

me = PathWeightSampling.MarginalEnsemble(sds)
ce = PathWeightSampling.ConditionalEnsemble(sds)

@test ce.dist === me.dist === sds.dist
@test ce.indep_idxs == [2, 3]
@test ce.index_map == [1]
@test ce.dtimes == me.dtimes == sds.dtimes

new_conf = PathWeightSampling.sample(conf, me)

@test conf.s_traj != new_conf.s_traj
@test conf.x_traj == new_conf.x_traj

alg = SMCEstimate(128)
mi = PathWeightSampling.mutual_information(sds, alg)

# Simpler SDEDrivenSystem
xn = @reaction_network GeneExpression begin
    ρ, L --> L + X
    μ, X --> ∅
end ρ μ

sds = PathWeightSampling.SDEDrivenSystem(
    sn, xn,
    [50, 50], # u0
    ps,
    [10.0, 10.0],  # px
    0.0:0.1:2.0  # dtimes
)

mi = PathWeightSampling.mutual_information(sds, alg)

# using Plots
# conf = PathWeightSampling.generate_configuration(sds, seed=3)
# plot(conf)

# me = PathWeightSampling.MarginalEnsemble(sds)

# x = PathWeightSampling.trajectory_energy(
#     me.dist,
#     PathWeightSampling.merge_trajectories(conf.s_traj, conf.x_traj)
# )
# -PathWeightSampling.energy_difference(conf, me)[end]
# plot(-PathWeightSampling.energy_difference(conf, me))

# samples = [PathWeightSampling.sample(conf, me) for i = 1:1024]
# log_probs = hcat(map(c -> -PathWeightSampling.energy_difference(c, me), samples)...)

# log_prob = mean(log_probs, dims=2)

# plot(-PathWeightSampling.energy_difference(conf, me) - log_prob)

# conf.s_traj
# merged = PathWeightSampling.collect_trajectory(PathWeightSampling.merge_trajectories(conf.s_traj, conf.x_traj))
# plot(conf)
# plot(merged)

# traj = PathWeightSampling.propagate(conf, me, [50], (0.0, 0.1))
# plot(traj)

# alg = SMCEstimate(128)
# mi = PathWeightSampling.mutual_information(sds, alg, num_samples=100)

# using Statistics
# plot(sds.dtimes, mean(mi.MutualInformation))
# plot!(sds.dtimes, 1.0 .* sds.dtimes)
