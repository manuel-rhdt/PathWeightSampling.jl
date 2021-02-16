import DiffEqJump


struct SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}
    integrator::DiffEqJump.SSAIntegrator{F,uType,tType,P,S,CB,SA,OPT,TS}
end

Base.IteratorSize(::Type{SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}}) where {F,uType,tType,P,S,CB,SA,OPT,TS} = Base.SizeUnknown()
Base.eltype(::Type{SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}}) where {F,uType,tType,P,S,CB,SA,OPT,TS} = Tuple{uType,tType}

function Base.iterate(iter::SSAIter)
    integrator = iter.integrator
    (integrator.u, integrator.t), nothing
end

function Base.iterate(iter::SSAIter, state::Nothing)
    integrator = iter.integrator
    if DiffEqJump.should_continue_solve(integrator)
        step!(integrator)
        return (integrator.u, integrator.t), nothing
    end
    
    end_time = integrator.sol.prob.tspan[2]
    if integrator.t < end_time
        integrator.t = end_time
        return (integrator.u, integrator.t), nothing
    end

    nothing
end

# An iterator over events (u, t) where t represents time and u the 
# state. We assume the trajectory has value u_i in the time span
# [t_i, t_{i+1})

struct EventThinner{I}
    inner::I
end

Base.IteratorSize(::Type{EventThinner{I}}) where I = Base.SizeUnknown()
Base.eltype(::Type{EventThinner{I}}) where I = eltype(I)

function Base.iterate(iter::EventThinner)
    result = iterate(iter.inner)
    if result === nothing
        nothing
    else
        ((u, t), state) = result
        ((u, t), (state, (copy(u), t)))
    end
end

function Base.iterate(iter::EventThinner, state::Tuple{<:Any,<:Any})
    (inner_state, (uprev, tprev)) = state
    if tprev == Inf
        return nothing
    end

    previous_time = tprev

    while true
        result = iterate(iter.inner, inner_state)
        if result === nothing
            if tprev != previous_time
                # return the last element as the final point of the trajectory
                return (uprev, tprev), (inner_state, (uprev, Inf))
            else
                return nothing
            end
        else
            ((u, t), inner_state) = result
            if u == uprev
                tprev = t
                continue
            end

            return ((u, t), (inner_state, (copy(u), t)))
        end
    end
end

function sub_trajectory(traj, indices)
    EventThinner((u[indices], t) for (u, t) in traj)
end


struct MergeIter{T1,T2}
    first::T1
    second::T2
end

Base.IteratorSize(::Type{MergeIter{T1,T2}}) where {T1,T2} = Base.SizeUnknown()
# Base.eltype(::Type{MergeIter{T1, T2}}) where {T1, T2} = eltype(T1)

function Base.iterate(iter::MergeIter)
    (u1, t1), state1 = iterate(iter.first)
    (u2, t2), state2 = iterate(iter.second)
    
    active_index = t1 <= t2
    t_active = active_index ? t1 : t2
    t_inactive = (!active_index) ? t1 : t2

    state = MergeState(
        state1, state2,
        u1, u2, u1, u2,
        active_index,
        t_inactive
    )

    while t_active <= t_inactive
        new_event = execute_event!(t_active, state)
        t_active = get_next_event!(iter, state)
    end

    new_event = (vcat(state.u1, state.u2), state.t_inactive)

    state.active_index = !state.active_index
    state.t_inactive = t_active

    new_event, state
end

mutable struct MergeState{S1,S2,U1,U2,tType <: Real}
    state1::S1
    state2::S2
    u1::U1
    u2::U2
    u1_next::U1
    u2_next::U2
    active_index::Bool
t_inactive::tType
end

function Base.iterate(iter::MergeIter, state::MergeState)
    # This code is carefully arranged such that all tests pass.

    if state.t_inactive == Inf
        return nothing
    end

    t_active = get_next_event!(iter, state)

    if t_active == Inf
        return nothing
    end

    if t_active > state.t_inactive
        state.active_index = !state.active_index 
        t_swap = t_active
        t_active = state.t_inactive
        state.t_inactive = t_swap
    end
    new_event = execute_event!(t_active, state)
    if state.t_inactive == t_active
        state.t_inactive = get_next_event!(iter, state)
        state.active_index = !state.active_index
        new_event = execute_event!(t_active, state)
    end

    return new_event, state
end

function get_next_event!(iter::MergeIter, state::MergeState)
    if state.active_index
        iter_result = iterate(iter.first, state.state1)
        if iter_result === nothing
            return Inf
        end
        (u, t), inner_state = iter_result
        state.state1 = inner_state
        state.u1_next = u
        return t
    else
        iter_result = iterate(iter.second, state.state2)
        if iter_result === nothing
            return Inf
        end
        (u, t), inner_state = iter_result
        state.state2 = inner_state
        state.u2_next = u
        return t
    end
end

function execute_event!(t, state::MergeState)
    if state.active_index
        state.u1 = state.u1_next
    else
        state.u2 = state.u2_next
    end
    u = vcat(state.u1, state.u2)
    (u, t)
end

# merge more than 2 trajectories using recursion
merge_trajectories(traj) = traj
function merge_trajectories(traj1, traj2)
    MergeIter(traj1, traj2)
end
function merge_trajectories(traj1, traj2, other_trajs...)
    merge12 = merge_trajectories(traj1, traj2)
    merge_trajectories(merge12, other_trajs...)
end

include("trajectories/trajectory.jl")


function collect_trajectory(iter)
    ((u, t), state) = iterate(iter)
    traj = Trajectory([t], [u])

    for (u, t) in Iterators.rest(iter, state)
        push!(traj.t, t)
        push!(traj.u, u)
    end

    traj
end
