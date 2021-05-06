import DiffEqJump
using Transducers
using Transducers: @next, complete, R_, inner, wrapping, xform, start, next, wrap, unwrap, Unseen

struct SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}
    integrator::DiffEqJump.SSAIntegrator{F,uType,tType,P,S,CB,SA,OPT,TS}
end

Base.IteratorSize(::Type{SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}}) where {F,uType,tType,P,S,CB,SA,OPT,TS} = Base.SizeUnknown()
Base.eltype(::Type{SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}}) where {F,uType,tType,P,S,CB,SA,OPT,TS} = Tuple{uType,tType,Int}

function Base.iterate(iter::SSAIter)
    integrator = iter.integrator
    end_time = integrator.sol.prob.tspan[2]
    if integrator.t >= end_time
        return nothing
    end
    aggregator = integrator.cb.condition
    next_jump_time = integrator.tstop > integrator.t ? integrator.tstop : typemax(integrator.tstop)
    if !isempty(integrator.tstops) &&
            integrator.tstops_idx <= length(integrator.tstops) &&
            integrator.tstops[integrator.tstops_idx] < next_jump_time
        next_jump_time = integrator.tstops[integrator.tstops_idx]
    end
    t = min(next_jump_time, end_time)
    i = t == end_time ? 0 : aggregator.next_jump
    return (integrator.u, t, i), ()
end

function Base.iterate(iter::SSAIter, state::Tuple{})
    integrator = iter.integrator
    if DiffEqJump.should_continue_solve(integrator)
        end_time = integrator.sol.prob.tspan[2]
        step!(integrator)
        aggregator = integrator.cb.condition
        i = aggregator.next_jump

        next_jump_time = integrator.tstop > integrator.t ? integrator.tstop : typemax(integrator.tstop)
        if !isempty(integrator.tstops) &&
                integrator.tstops_idx <= length(integrator.tstops) &&
                integrator.tstops[integrator.tstops_idx] < next_jump_time
            next_jump_time = integrator.tstops[integrator.tstops_idx]
            i = 0
        end
        t = min(next_jump_time, end_time)
        if t == end_time 
            i = 0
        end

        return (integrator.u, t, i), ()
    end
    nothing
end

function sub_trajectory(traj, indices)
    traj |> Map((u,t,i)::Tuple -> ((@view u[indices]), t, i)) |> Thin() |> collect_trajectory
end
struct Chain{V,T <: AbstractVector,U <: AbstractVector} <: AbstractVector{V}
    head::T
    tail::U

    function Chain(head::AbstractVector{X}, tail::AbstractVector{Y}) where {X, Y} 
        new{promote_type(X, Y), typeof(head), typeof(tail)}(head, tail)
    end
end

Base.IndexStyle(::Type{<:Chain}) = IndexLinear()
Base.size(ch::Chain) = size(ch.head) .+ size(ch.tail)
Base.getindex(ch::Chain{V}, i::Int) where {V} = convert(V, i > length(ch.head) ? ch.tail[i - length(ch.head)] : ch.head[i])
Base.setindex!(ch::Chain, v, i::Int) = i > length(ch.head) ? ch.tail[i - length(ch.head)] = v : ch.head[i] = v

# merge more than 2 trajectories using recursion
merge_trajectories(traj) = traj
function merge_trajectories(traj1, traj2)
    traj1 |> MergeWith(traj2)
end
function merge_trajectories(traj1, traj2, other_trajs...)
    merge12 = merge_trajectories(traj1, traj2)
    merge_trajectories(merge12, other_trajs...)
end

struct Thin <: Transducers.AbstractFilter end

Transducers.start(rf::R_{Thin}, result) = wrap(rf, Unseen(), start(inner(rf), result))

function Transducers.complete(rf::R_{Thin}, result)
    result = wrapping(rf, result) do prev, iresult
        if prev isa Unseen
            return prev, iresult
        end
        return prev, next(inner(rf), iresult, prev)
    end
    complete(inner(rf), unwrap(rf, result)[2])
end

@inline Transducers.next(rf::R_{Thin}, result, (u, t, i)) =
    wrapping(rf, result) do prev, iresult
        if prev isa Unseen
            return (copy(u), t, i), iresult
        end
        if prev[1] != u
            return (copy(u), t, i), next(inner(rf), iresult, prev)
        else
            return (prev[1], t, i), iresult
        end
    end

struct MergeWith{uType,tType} <: Transducer
    traj::Trajectory{uType,tType}
    start_index::Int
end

MergeWith(traj::Trajectory) = MergeWith(traj, 1)

function Transducers.start(rf::R_{MergeWith}, result)
    private_state = xform(rf).start_index
    return wrap(rf, private_state, start(inner(rf), result))
end

function Transducers.next(rf::R_{MergeWith}, result, (u, t, i))
    f = let u=u, t=t, i=i, rf=rf
        function (index::Int, iresult)
            merge_traj = xform(rf).traj
            ri = i
            @inbounds while index <= length(merge_traj.t) && merge_traj.t[index] < t
                merge_i = merge_traj.i[index]
                iresult = next(inner(rf), iresult, (Chain(u, merge_traj.u[index]), merge_traj.t[index], merge_i))
                index += 1
            end

            if index > length(merge_traj.t)
                return index, reduced(iresult)
            end

            iresult = next(inner(rf), iresult, (Chain(u, (@inbounds merge_traj.u[index])), t, ri))

            @inbounds if index <= length(merge_traj.t) && merge_traj.t[index] == t
                index += 1
            end

            index, iresult
        end
    end

    wrapping(f, rf, result)
end

function Transducers.complete(rf::R_{MergeWith}, result)
    _private_state, inner_result = unwrap(rf, result)
    return complete(inner(rf), inner_result)
end

function collect_trajectory(iter)
    ((u, t, i), state) = iterate(iter)
    traj = Trajectory([t], [copy(u)], Int[i])

    for (u, t, i) in Iterators.rest(iter, state)
        push!(traj.t, t)
        push!(traj.u, copy(u))
        push!(traj.i, i)
    end

    traj
end

function collect_trajectory(xf::Transducers.Transducer, itr)
    rf = Transducers.ProductRF(Map(copy)'(Transducers.push!!), Transducers.push!!, Transducers.push!!) 
    (u, t, i) = foldxl(rf, xf, itr; init=(Transducers.Empty(Vector), Transducers.Empty(Vector), Transducers.Empty(Vector)))
    Trajectory(t, u, i)
end

collect_trajectory(ed::Transducers.Eduction) = collect_trajectory(Transducers.extract_transducer(ed)...)
