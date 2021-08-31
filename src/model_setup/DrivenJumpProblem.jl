
module DrivenJumpProblems

export DrivenJumpProblem

using DiffEqJump
using CommonSolve

struct IdentityMap end
@inline Base.getindex(x::IdentityMap, i::Integer) = i

mutable struct TrajectoryCallback{Trajectory, IndexMap}
    traj::Trajectory
    index::Int
    index_map::IndexMap
end

TrajectoryCallback(traj, index_map = IdentityMap()) = TrajectoryCallback(traj, 1, index_map)

function (tc::TrajectoryCallback)(integrator::DiffEqBase.DEIntegrator) # affect!
    traj = tc.traj
    tc.index = min(tc.index + 1, length(traj.t))
    cond_u = traj.u[tc.index-1]
    for i in eachindex(cond_u)
        integrator.u[tc.index_map[i]] = cond_u[i]
    end
    # it is important to call this to properly update reaction rates
    DiffEqJump.reset_aggregated_jumps!(integrator, nothing, integrator.cb)
    nothing
end

function (tc::TrajectoryCallback)(u, t::Real, i::DiffEqBase.DEIntegrator)::Bool # condition
    @inbounds tcb = tc.traj.t[tc.index]
    while tc.index < length(tc.traj.t) && t > tcb
        tc.index += 1
        @inbounds tcb = tc.traj.t[tc.index]
    end
    t == tcb
end

"""
    DrivenJumpProblem(jump_problem, driving_trajectory, index_map = IdentityMap())

Create a `DrivenJumpProblem` from a base `jump_problem` and a `driving_trajectory`.

The optional argument `index_map` specifies how the components from `driving_trajectory`
map onto the components of the `jump_problem`. The default `IdentityMap` maps the N components
of the `driving_trajectory` onto the first N components of the jump problem, leaving the 
remaining components of the jump problem unaltered by the driving force.
"""
struct DrivenJumpProblem{Prob,Cb}
    prob::Prob
    callback::Cb

    function DrivenJumpProblem(jump_problem::JP, driving_trajectory, index_map = IdentityMap()) where {JP}
        tcb = TrajectoryCallback(driving_trajectory, index_map)
        callback = DiscreteCallback(tcb, tcb, save_positions=(false, false))
        new{JP, typeof(callback)}(jump_problem, callback)
    end
end

function CommonSolve.init(prob::DrivenJumpProblem)
    prob.callback.condition.index = 1
    tstops = prob.callback.condition.traj.t
    from = searchsortedfirst(tstops, prob.prob.prob.tspan[1])
    to = searchsortedlast(tstops, prob.prob.prob.tspan[2])
    DiffEqBase.init(prob.prob, SSAStepper(), callback=prob.callback, tstops=tstops[from:to])
end

function CommonSolve.solve(prob::DrivenJumpProblem)
    integrator = init(prob)
    solve!(integrator)
    integrator.sol
end

Base.summary(io::IO, prob::DrivenJumpProblem) = string(DiffEqBase.parameterless_type(prob)," with problem ",DiffEqBase.parameterless_type(prob.prob))
function Base.show(io::IO, mime::MIME"text/plain", A::DrivenJumpProblem)
  println(io,summary(A))
#   println(io,"Number of constant rate jumps: ",A.discrete_jump_aggregation === nothing ? 0 : num_constant_rate_jumps(A.discrete_jump_aggregation))
#   println(io,"Number of variable rate jumps: ",length(A.variable_jumps))
#   if A.regular_jump !== nothing
    # println(io,"Have a regular jump")
#   end
#   if (A.massaction_jump !== nothing) && (get_num_majumps(A.massaction_jump) > 0)
    # println(io,"Have a mass action jump")
#   end
end

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