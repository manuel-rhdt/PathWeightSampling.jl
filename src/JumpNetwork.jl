using Catalyst
using ModelingToolkit
using DiffEqJump
using DiffEqBase
using StaticArrays
using GaussianMcmc

import DiffEqCallbacks: PresetTimeCallback
import Distributions: logpdf

mutable struct ConditionalEnsemble{JP, CB}
    jump_problem::JP
    cond_callback::CB
    cond_traj::GaussianMcmc.Trajectory
end

function ConditionalEnsemble(
    network::ReactionSystem, 
    cond_traj::GaussianMcmc.Trajectory,
    u0::AbstractVector{T},
    p
) where T
    function affect!(integrator)
        integrator.u = setindex(integrator.u, cond_traj(integrator.t)[1], 1)
        # it is important to call this to properly update reaction rates
        DiffEqJump.reset_aggregated_jumps!(integrator)
    end
    callback = DiscreteCallback((u,t,i) -> t ∈ traj.t, affect!, save_positions=(false, false))

    dprob = DiscreteProblem(network, u0, tspan, p)
    # we have to remake the discrete problem with u0 as StaticArray for 
    # improved performance
    dprob = remake(dprob, u0=SVector{length(u0), T}(u0))
    jprob = JumpProblem(network, dprob, Direct(), save_positions=(false, false))

    ConditionalEnsemble(jprob, callback, cond_traj)
end

function sample(ensemble::ConditionalEnsemble)
    sol = solve(ensemble.jump_problem, SSAStepper(), callback=ensemble.cond_callback, tstops=ensemble.cond_traj.t)
    GaussianMcmc.Trajectory(sol)
end

function init(ensemble::ConditionalEnsemble)
    DiffEqBase.init(ensemble.jump_problem, SSAStepper(), callback=ensemble.cond_callback, tstops=ensemble.cond_traj.t)
end

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
    ρ, L + R --> L + LR
    μ, LR --> R
    ξ, R + CheY --> R + CheYp
    ν, CheYp --> CheY
end ρ μ ξ ν


joint = merge(sn, rn)

using Plots

u0 = SA[10, 30, 0, 50, 0]
tspan = (0.0, 10.0)
p = [5.0, 1.0, 1.0, 4.0, 1.0, 2.0]
dprob = DiscreteProblem(joint, u0, tspan, p)
dprob = remake(dprob, u0=u0)
jprob = JumpProblem(joint, dprob, Direct())
sol = solve(jprob, SSAStepper())
plot(sol)

traj = GaussianMcmc.Trajectory(GaussianMcmc.trajectory(sol, [1]))

ensemble = ConditionalEnsemble(rn, traj, [10, 30, 0, 50, 0], [1.0, 4.0, 1.0, 2.0])

integrator = init(ensemble)
iterator = GaussianMcmc.SSAIter(integrator)
subtraj = GaussianMcmc.sub_trajectory(iterator, [4])
GaussianMcmc.collect_trajectory(subtraj)

list = []
for i in 1:2000
    step!(integrator)
    push!(list, integrator.t => integrator.u)
end

for x in tuples(integrator)
    push!(list, x)
end

gen = ((u[1], t) for (u, t) ∈ tuples(integrator))

for x in gen
    push!(list, x)
end

step!(integrator)

integrator.keep_stepping

list

list

y = hcat(Vector.(getindex.(list, 2))...)
plot(getindex.(list, 1), y', linetype=:steppost)

plot(getindex.(list, 1), (integrator.sol[:,3:end-6] - y)')

plot(integrator.sol)


dist = GaussianMcmc.distribution(rn)

ftraj = sample(ensemble)
samples = [logpdf(dist, sample(ensemble), params=p) for i in 1:1000]

histogram(samples)

plot(traj)

xn = @reaction_network begin
    δ, CheYp --> CheYp + X
    χ, x --> ∅
end δ χ

GaussianMcmc.distribution(xn)