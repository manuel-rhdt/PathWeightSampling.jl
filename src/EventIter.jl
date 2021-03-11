import DiffEqJump


struct SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}
    integrator::DiffEqJump.SSAIntegrator{F,uType,tType,P,S,CB,SA,OPT,TS}
end

Base.IteratorSize(::Type{SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}}) where {F,uType,tType,P,S,CB,SA,OPT,TS} = Base.SizeUnknown()
Base.eltype(::Type{SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}}) where {F,uType,tType,P,S,CB,SA,OPT,TS} = Tuple{uType,tType,Int}

function Base.iterate(iter::SSAIter)
    integrator = iter.integrator
    (integrator.u, integrator.t, 0), nothing
end

function Base.iterate(iter::SSAIter, state::Nothing)
    integrator = iter.integrator
    if DiffEqJump.should_continue_solve(integrator)
        step!(integrator)
        aggregator = integrator.cb.condition
        return (integrator.u, integrator.t, aggregator.prev_jump), nothing
    end
    
    end_time = integrator.sol.prob.tspan[2]
    if integrator.t < end_time
        integrator.t = end_time
        return (integrator.u, integrator.t, 0), nothing
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
        ((u, t, i), state) = result
        ((u, t, i), (state, (u, t, i)))
    end
end

function Base.iterate(iter::EventThinner, state::Tuple{<:Any,<:Any})
    (inner_state, (uprev, tprev, iprev)) = state
    if tprev == Inf
        return nothing
    end

    previous_time = tprev

    while true
        result = iterate(iter.inner, inner_state)
        if result === nothing
            if tprev != previous_time
                # return the last element as the final point of the trajectory
                return (uprev, tprev, iprev), (inner_state, (uprev, Inf, iprev))
            else
                return nothing
            end
        else
            ((u, t, i), inner_state) = result
            if u == uprev
                tprev = t
                continue
            end

            return ((u, t, i), (inner_state, (u, t, i)))
        end
    end
end

function sub_trajectory(traj, indices)
    EventThinner(((@view u[indices]), t, i) for (u, t, i) in traj)
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
        0, 0,
        active_index,
        t_inactive
    )

    while t_active <= t_inactive
        new_event, state = execute_event(t_active, state)
        t_active, state = get_next_event(iter, state)
    end

    new_event = (Chain(state.u1, state.u2), state.t_inactive, 0)

    state = setfield(state, active_index=!state.active_index, t_inactive=t_active)

    new_event, state
end

struct MergeState{S1,S2,U1,U2,tType <: Real}
    state1::S1
    state2::S2
    u1::U1
    u2::U2
    u1_next::U1
    u2_next::U2
    i1_next::Int
    i2_next::Int
    active_index::Bool
t_inactive::tType
end

setfield(s::MS;
    state1=s.state1, 
    state2=s.state2,
    u1=s.u1,
    u2=s.u2,
    u1_next=s.u1_next,
    u2_next=s.u2_next,
    i1_next=s.i1_next,
    i2_next=s.i2_next,
    active_index=s.active_index,
    t_inactive=s.t_inactive) where {MS <: MergeState} = MergeState(state1, state2, u1, u2, u1_next, u2_next, i1_next, i2_next, active_index, t_inactive)

function Base.iterate(iter::MergeIter, state::MergeState)
    # This code is carefully arranged such that all tests pass.

    if state.t_inactive == Inf
        return nothing
    end

    t_active, state = get_next_event(iter, state)

    if t_active == Inf
        return nothing
    end

    if t_active > state.t_inactive
        t_swap = t_active
        t_active = state.t_inactive
        state = setfield(state, active_index=!state.active_index, t_inactive=t_swap)
    end
    new_event, state = execute_event(t_active, state)
    if state.t_inactive == t_active
        t, state = get_next_event(iter, state)
        state = setfield(state, active_index=!state.active_index, t_inactive=t)
        new_event, state = execute_event(t_active, state)
    end

    return new_event, state
end

function get_next_event(iter::MergeIter, state::MergeState)
    if state.active_index
        iter_result = iterate(iter.first, state.state1)
        if iter_result === nothing
            return (Inf, state)
        end
        (u, t, i), inner_state = iter_result
        state = setfield(state, state1=inner_state, u1_next=u, i1_next=i)
        return (t, state)
    else
        iter_result = iterate(iter.second, state.state2)
        if iter_result === nothing
            return (Inf, state)
        end
        (u, t, i), inner_state = iter_result
        state = setfield(state, state2=inner_state, u2_next=u, i2_next=i)
        return (t, state)
    end
end

function execute_event(t, state::MergeState)
    i = 0
    if state.active_index
        i = state.i1_next
        state = setfield(state, u1=state.u1_next, i1_next=0)
    else
        i = state.i2_next
        state = setfield(state, u2=state.u2_next, i2_next=0)
    end
    u = Chain(state.u1, state.u2)
    (u, t, i), state
end

struct Chain{U,T <: AbstractVector{U},V <: AbstractVector{U}} <: AbstractVector{U}
    head::T
    tail::V
end

Base.IndexStyle(::Type{<:Chain}) = IndexLinear()
Base.size(ch::Chain) = size(ch.head) .+ size(ch.tail)
Base.getindex(ch::Chain, i::Int) = i > length(ch.head) ? ch.tail[i - length(ch.head)] : ch.head[i]
Base.setindex!(ch::Chain, v, i::Int) = i > length(ch.head) ? ch.tail[i - length(ch.head)] = v : ch.head[i] = v

# merge more than 2 trajectories using recursion
merge_trajectories(traj) = traj
function merge_trajectories(traj1, traj2)
    MergeIter(traj1, traj2)
end
function merge_trajectories(traj1, traj2, other_trajs...)
    merge12 = merge_trajectories(traj1, traj2)
    merge_trajectories(merge12, other_trajs...)
end

function collect_trajectory(iter)
    ((u, t, i), state) = iterate(iter)
    traj = Trajectory([t], [copy(u)], Int[])

    for (u, t, i) in Iterators.rest(iter, state)
        push!(traj.t, t)
        push!(traj.u, copy(u))
        push!(traj.i, i)
    end

    if length(traj.i) > 0
        # remove last reaction index at end of trajectory
        pop!(traj.i)
    end

    traj
end
