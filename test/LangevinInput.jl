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