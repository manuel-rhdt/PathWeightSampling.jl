using Catalyst
using ModelingToolkit
using DiffEqJump
using DiffEqBase
using StaticArrays
using GaussianMcmc

import DiffEqCallbacks: PresetTimeCallback
import Distributions: logpdf
import GaussianMcmc: SSAIter, collect_trajectory, sub_trajectory, merge_trajectories

mutable struct ConditionalEnsemble{JP,CB}
    jump_problem::JP
    cond_callback::CB
    cond_traj::GaussianMcmc.Trajectory
end

function ConditionalEnsemble(
    network::ReactionSystem, 
    cond_traj::GaussianMcmc.Trajectory,
    u0::AbstractVector{T},
    p,
    tspan
) where T
    affect! = function (integrator)
        cond_u = cond_traj(integrator.t)
        for i in eachindex(cond_u)
            integrator.u = setindex(integrator.u, cond_u[i], i)
        end
        # it is important to call this to properly update reaction rates
        DiffEqJump.reset_aggregated_jumps!(integrator, nothing, integrator.cb)
    end
    callback = DiscreteCallback((u, t, i) -> t ∈ cond_traj.t, affect!, save_positions=(false, false))

    dprob = DiscreteProblem(network, u0, tspan, p)
    # we have to remake the discrete problem with u0 as StaticArray for 
    # improved performance
    dprob = remake(dprob, u0=SVector{length(u0),T}(u0))
    jprob = JumpProblem(network, dprob, Direct(), save_positions=(false, false))

    ConditionalEnsemble(jprob, callback, cond_traj)
end

function init(ensemble::ConditionalEnsemble)
    DiffEqBase.init(ensemble.jump_problem, SSAStepper(), callback=ensemble.cond_callback, tstops=ensemble.cond_traj.t)
end

ssa_iter(ensemble::ConditionalEnsemble) = SSAIter(init(ensemble))

struct SRXsystem

end

function generate_configuration(SRXsystem)
    # we first generate a joint SRX trajectory

    u0 = SA[10, 30, 0, 50, 0, 0]
    tspan = (0.0, 10.0)
    p = [5.0, 1.0, 1.0, 4.0, 1.0, 2.0, 1.0, 1.0]
    dprob = DiscreteProblem(joint, u0, tspan, p)
    dprob = remake(dprob, u0=u0)
    jprob = JumpProblem(joint, dprob, Direct())

    sol = solve(jprob, SSAStepper())

    # then we create the R ensemble conditioned on the signal
    signal = GaussianMcmc.Trajectory(GaussianMcmc.trajectory(sol, [1]))
    r_ensemble = ConditionalEnsemble(rn, signal, [10, 30, 0, 50, 0], [1.0, 4.0, 1.0, 2.0], tspan)

    # finally we extract the X part from the SRX trajectory
    x_traj = GaussianMcmc.Trajectory(GaussianMcmc.trajectory(sol, [6]))
    x_dist = distribution(xn)

    SRXconfiguration(r_ensemble, SA[5], x_traj, x_dist)
end

struct SRXconfiguration{RE,RIdx,XT,XD}
    r_ensemble::RE
    r_idxs::RIdx
    x_traj::XT
    x_dist::XD
end

function sample(configuration::SRXconfiguration; θ=0.0)
    if θ!=0.0
        error("can only use DirectMC with JumpNetwork")
    end
    collect_trajectory(sub_trajectory(ssa_iter(jump_newtork.r_ensemble), jump_newtork.r_idxs))
end

function energy_difference(r_traj, configuration::SRXconfiguration, params)
    -logpdf(jump_network.x_dist, merge_trajectories(r_traj, jump_network.x_traj), params=params)
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

xn = @reaction_network begin
    δ, CheYp --> CheYp + X
    χ, X --> ∅
end δ χ

joint = merge(merge(sn, rn), xn)

using Plots

u0 = SA[10, 30, 0, 50, 0, 0]
tspan = (0.0, 10.0)
p = [5.0, 1.0, 1.0, 4.0, 1.0, 2.0, 1.0, 1.0]
dprob = DiscreteProblem(joint, u0, tspan, p)
dprob = remake(dprob, u0=u0)
jprob = JumpProblem(joint, dprob, Direct())
sol = solve(jprob, SSAStepper())
plot(sol)

traj = GaussianMcmc.Trajectory(GaussianMcmc.trajectory(sol, [1]))
x_traj = GaussianMcmc.Trajectory(GaussianMcmc.trajectory(sol, [6]))
x_dist = GaussianMcmc.distribution(xn)

ensemble = ConditionalEnsemble(rn, traj, [10, 30, 0, 50, 0], [1.0, 4.0, 1.0, 2.0], tspan)

jn = JumpNetwork(ensemble, SA[5], x_traj, x_dist)

get_sample = function ()
    r_traj = sample(jn)
    energy_difference(r_traj, jn, SA[1.0, 1.0])
end

histogram([get_sample() for i=1:10000])

iterator = ssa_iter(ensemble)
subtraj = GaussianMcmc.sub_trajectory(iterator, SA[5])
subtraj = collect_trajectory(subtraj)

x_ens = ConditionalEnsemble(xn, subtraj, [1, 0], [1.0, 1.0], tspan)

plot(collect_trajectory(ssa_iter(x_ens)))

results = [logpdf(dist, ssa_iter(x_ens), params=[1.0, 1.0]) for i = 1:10000]

histogram(results)
