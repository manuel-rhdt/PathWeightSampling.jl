using Catalyst
using ModelingToolkit
using DiffEqJump
using DiffEqBase
using StaticArrays
using GaussianMcmc



# To compute the MI we need the ability
# - create a new configuration (i.e. jointly sample S, R, X)
# - for a given configuration replace the R part of the trajectory
# - for a given configuration replace the S part of the trajectory
# - compute P(r, x | s)
# - compute P_0(r)

sn = @reaction_network begin
    κ, ∅ --> 2L
    λ, L --> ∅
end κ λ

rn = @reaction_network begin
    ρ, L --> L + LR
    μ, LR --> ∅
    ξ, LR + CheY --> LR + CheYp
    ν, CheYp --> CheY
end ρ μ ξ ν


joint = merge(sn, rn)

using Plots

u0 = SA[10, 0, 50, 0]
tspan = (0.0, 10.0)
p = [50.0, 1.0, 5*10.0, 5*20.0, 1.0, 2.0]
dprob = DiscreteProblem(joint, u0, tspan, p)
dprob = remake(dprob, u0=u0)
jprob = JumpProblem(joint, dprob, Direct())
sol = solve(jprob, SSAStepper())
plot(sol)

integrator = init(jprob, SSAStepper(), opts=(tstops=[]))

for x in Iterators.take(integrator, 10)
    @show x
end

integrator.sol

xn = @reaction_network begin
    δ, ∅ --> S
    χ, S --> ∅
end κ λ