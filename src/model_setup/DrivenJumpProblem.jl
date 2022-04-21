
module DrivenJumpProblems

export DrivenJumpProblem

using DiffEqJump
using SciMLBase
using PathWeightSampling: Trajectory
using StaticArrays
using StochasticDiffEq

"""
    IdentityMap()

A simple helper type that always returns `i` when indexed as `IdentityMap()[i]`.
"""
struct IdentityMap end
@inline Base.getindex(x::IdentityMap, i::Integer) = i

mutable struct TrajectoryCallback{T,IndexMap}
    traj::T
    index::Int
    index_map::IndexMap
end

TrajectoryCallback(traj, index_map=IdentityMap()) = TrajectoryCallback(traj, 1, index_map)

function (tc::TrajectoryCallback)(integrator::DiffEqBase.DEIntegrator) # affect!
    traj = tc.traj
    tc.index = min(tc.index + 1, length(traj.t))
    cond_u = traj.u[tc.index]
    for i in eachindex(cond_u)
        if integrator.u isa StaticVector
            integrator.u = setindex(integrator.u, cond_u[i], tc.index_map[i])
        else
            integrator.u[tc.index_map[i]] = cond_u[i]
        end
    end
    # it is important to call this to properly update reaction rates
    DiffEqJump.reset_aggregated_jumps!(integrator)
    nothing
end

function (tc::TrajectoryCallback)(u, t::Real, i::DiffEqBase.DEIntegrator)::Bool # condition
    tc_index = tc.index
    traj_t = tc.traj.t
    traj_len = length(traj_t)
    @inbounds tcb = traj_t[tc_index]
    while tc_index < traj_len && t > tcb
        tc_index += 1
        @inbounds tcb = traj_t[tc_index]
    end
    tc.index = tc_index
    t == tcb
end

"""
    DrivenJumpProblem(jump_problem, driving_trajectory; index_map = IdentityMap(), save_jumps=false)

Create a `DrivenJumpProblem` from a base `jump_problem` and a `driving_trajectory`.

The optional argument `index_map` specifies how the components from `driving_trajectory`
map onto the components of the `jump_problem`. The default `IdentityMap` maps the N components
of the `driving_trajectory` onto the first N components of the jump problem, leaving the 
remaining components of the jump problem unaltered by the driving trajectory.
"""
struct DrivenJumpProblem{Prob,Cb,TG}
    prob::Prob
    callback::Cb
    make_trajectory::TG
end

function DrivenJumpProblem(jump_problem::JP, driving_trajectory::Trajectory; index_map=IdentityMap(), save_jumps=false) where {JP}
    tcb = TrajectoryCallback(driving_trajectory, index_map)
    callback = DiscreteCallback(tcb, tcb, save_positions=(false, save_jumps))
    function make_trajectory(prob::DrivenJumpProblem)
        driving_trajectory
    end
    DrivenJumpProblem(jump_problem, callback, make_trajectory)
end

function DrivenJumpProblem(jump_problem::JP, driving_problem::SDEProblem; index_map=IdentityMap(), save_jumps=false) where {JP}
    tcb = TrajectoryCallback(Trajectory(Vector{Float64}[], Float64[]), index_map)
    callback = DiscreteCallback(tcb, tcb, save_positions=(false, save_jumps))
    function make_trajectory(prob::DrivenJumpProblem)
        u0 = prob.prob.prob.u0[begin:length(driving_problem.u0)]
        sde_prob = remake(driving_problem, u0=u0, tspan=prob.prob.prob.tspan)
        sol = solve(sde_prob, SOSRA(), saveat=0.01)
        Trajectory(sol.u[begin:end-1], sol.t[begin+1:end])
    end
    DrivenJumpProblem(jump_problem, callback, make_trajectory)
end

function SciMLBase.init(prob::DrivenJumpProblem; kwargs...)
    tspan = prob.prob.prob.tspan
    driving_trajectory = prob.make_trajectory(prob)

    prob.callback.condition.traj = driving_trajectory
    prob.callback.condition.index = 1

    # find tstops
    tstops = driving_trajectory.t
    from = searchsortedfirst(tstops, tspan[1])
    to = searchsortedlast(tstops, tspan[2])
    tstops_clipped = @view tstops[from:to]

    integrator = DiffEqJump.init(prob.prob, SSAStepper(), callback=prob.callback, tstops=tstops_clipped, save_start=false)
    integrator
end

function SciMLBase.solve(prob::DrivenJumpProblem; kwargs...)
    integrator = init(prob; kwargs...)
    solve!(integrator)
    integrator.sol
end

# for remaking
Base.@pure remaker_of(prob::T) where {T<:DrivenJumpProblem} = DiffEqBase.parameterless_type(T)
function DiffEqBase.remake(thing::DrivenJumpProblem; kwargs...)
    T = remaker_of(thing)

    errmesg = """
    DrivenJumpProblems can currently only be remade with new u0, p, tspan or prob fields. To change other fields create a new JumpProblem.
    """
    !issubset(keys(kwargs), (:u0, :p, :tspan, :prob)) && error(errmesg)

    if :prob ∉ keys(kwargs)
        jprob = DiffEqBase.remake(thing.prob; kwargs...)
    else
        any(k -> k in keys(kwargs), (:u0, :p, :tspan)) && error("If remaking a DrivenJumpProblem you can not pass both prob and any of u0, p, or tspan.")
        jprob = kwargs[:prob]
    end

    T(jprob, thing.callback, thing.make_trajectory)
end

Base.summary(io::IO, prob::DrivenJumpProblem) = string(DiffEqBase.parameterless_type(prob), " with problem ", DiffEqBase.parameterless_type(prob.prob))
function Base.show(io::IO, mime::MIME"text/plain", A::DrivenJumpProblem)
    println(io, summary(A))
end

# =======================================
# THE CODE BELOW IS CURRENTLY UNUSED
# =======================================

# We define a new jump aggregator to deal with a jump process that is driven
# by an input signal. The external driving signal is supplied as a trajectory of
# jump times t_i and values u_i. The reaction rates are determined by mass
# action kinetics.


function time_to_next_jump(remaining_input_trajectory)
    t_jump = zero(Float64)
    r = randexp()

    while true
        u_s, t_s = current_values(remaining_input_trajectory)
        u_s_next, t_s_next = iterate(remaining_input_trajectory)
        sumrate = eval_sumrate(u_s, u_x)
        next_sumrate = eval_sumrate(u_s_next, u_x)

        Δt = t_s_next - t_s
        if Δt * (sumrate + next_sumrate) / 2 > r
            # the next jump does not occur before t_s_next
            r -= Δt * (sumrate + next_sumrate) / 2
            continue
        else
            # now we have to find the precise jump time between t_s and t_s_next
            # which is the solution to the equation r = ∫_[t, t+Δt] sumrate(t) dt
            # when solved for Δt
            sumrate_slope = (next_sumrate - sumrate) / Δt
            t_jump = t_s + solve_quadratic(0.5 * sumrate_slope, sumrate, -r)[2]
            break
        end
    end

    t_jump
end

# ================== UTILITY FUNCTIONS ==================

# computes ab-cd in a numerically stable way
@inline function diff_of_products(a, b, c, d)
    w = d * c
    e = fma(-d, c, w)
    f = fma(a, b, -w)
    f + e
end

# computes real roots of quadratic equation ax^2 + bx + c = 0, assuming discriminant is positive
@inline function solve_quadratic(a, b, c)
    discriminant = sqrt(diff_of_products(b, b, 4 * a, c))
    q = -0.5 * (b + copysign(discriminant, b))
    q / a, c / q
end

end