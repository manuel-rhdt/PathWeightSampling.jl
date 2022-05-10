using RecipesBase
using DiffEqBase
import SciMLBase
import StaticArrays: SVector
using RecursiveArrayTools

abstract type AbstractTrajectory{U,T,A} <: AbstractVectorOfArray{U,2,A} end

"""
    Trajectory(u, t[, i])

A trajectory represented as a list of events.

# Fields

- `t`: Vector of event times.
- `u`: Vector of the same length as `t`. `u[i]` is the vector of copy numbers in the interval from `t[i-1]` to `t[i]`.
- `i`: A vector of event/reaction indices corresponding to the event times `t`. If empty, no event indices were recorded.

# Indexing

Indexing a trajectory with an integer index yields a tuple `(u, t, i)`. `u` is the trajectory value **before** the event at time `t`!
"""
struct Trajectory{U,T<:Real,A,B<:AbstractVector{T},C,D,E} <: AbstractTrajectory{U,T,A}
    u::VectorOfArray{U,2,A} # vector of length N. u[i] is the vector of copy numbers in the interval from t[i-1] to t[i]
    t::B # vector of length N, recording the reaction times and the end time of the trajectory
    i::C # vector of length N recording all reaction indices, or `nothing` if no reaction indices
    syms::D
    indepsym::E
end

Trajectory(u::VectorOfArray, t::Vector, i=nothing; syms=nothing, indepsym=nothing) = Trajectory(u, t, i, syms, indepsym)
Trajectory(u::Vector{<:AbstractVector}, t::Vector, i=nothing; syms=nothing, indepsym=nothing) = Trajectory(VectorOfArray(u), t, i; syms, indepsym)

function Base.copy(traj::Trajectory)
    Trajectory(
        copy(traj.u),
        deepcopy(traj.t),
        isnothing(traj.i) ? nothing : copy(traj.i),
        isnothing(traj.syms) ? nothing : copy(traj.syms),
        traj.indepsym
    )
end

Base.@propagate_inbounds Base.getindex(traj::AbstractTrajectory{U,T,A}, I::Int...) where {U,T,A} = traj.u[I[end]][Base.front(I)...]
Base.@propagate_inbounds Base.getindex(traj::AbstractTrajectory{U,T,A}, I::Int) where {U,T,A} = traj.u[I]
Base.@propagate_inbounds Base.getindex(traj::AbstractTrajectory{U,T,A}, I::Colon) where {U,T,A} = traj.u[I]
Base.@propagate_inbounds Base.getindex(traj::AbstractTrajectory{U,T,A}, I::AbstractArray{Int}) where {U,T,A} = Trajectory(traj.u[I], traj.t[I], isempty(traj.t) ? Int[] : traj.i[I])
# Base.@propagate_inbounds Base.getindex(traj::Trajectory, idx::Int) = (traj.u[idx], traj.t[idx], idx > length(traj.i) ? 0 : traj.i[idx])
Base.@propagate_inbounds function Base.getindex(traj::AbstractTrajectory{U,T,A},
    I::Union{Int,AbstractArray{Int},CartesianIndex,Colon,BitArray,AbstractArray{Bool}}...) where {U,T,A}
    traj.u[I...]
end

Base.@propagate_inbounds Base.getindex(t::AbstractTrajectory{U,T,A}, i::Int, ::Colon) where {U,T,A} = [t.u[j][i] for j in 1:length(t)]
Base.@propagate_inbounds Base.getindex(t::AbstractTrajectory{U,T,A}, ::Colon, i::Int) where {U,T,A} = t.u[i]
Base.@propagate_inbounds Base.getindex(t::AbstractTrajectory{U,T,A}, i::Int, II::AbstractArray{Int}) where {U,T,A} = [t.u[j][i] for j in II]

Base.@propagate_inbounds function Base.getindex(traj::AbstractTrajectory{U,T,A}, ii::CartesianIndex) where {U,T,A}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return traj.u[i][jj]
end

function equal_i(i1, i2)
    (i1 === nothing || isempty(i1)) && (i2 === nothing || isempty(i2)) || i1 == i2
end
Base.:(==)(traj1::AbstractTrajectory, traj2::AbstractTrajectory) = (traj1.t == traj2.t) && (traj1.u == traj2.u) && equal_i(traj1.i, traj2.i)

function (t::Trajectory)(time::Real)
    index = searchsortedfirst(t.t, time)
    if t.t[index] == time && index != lastindex(t.t)
        index += 1
    end
    if index > lastindex(t.t)
        error("Can't access trajectory that ends at t=$(last(t.t)) at time $time.")
    end
    t.u[index]
end

function (t::Trajectory)(times::AbstractArray{<:Real})
    hcat(t.(times)...)
end

function Trajectory(u::AbstractMatrix{T}, t::AbstractVector{tType}, i=nothing) where {tType<:Real,T<:Real}
    num_components = size(u, 1)
    if num_components > 0
        u_vec = [SVector{num_components}(c) for c in eachcol(u)]
    else
        u_vec = SVector{0,T}[]
    end

    Trajectory(u_vec, convert(Vector{tType}, t), i)
end

function Trajectory(t::Trajectory)
    t
end

function Trajectory(sol::ODESolution)
    Trajectory(sol.u[begin:end-1], sol.t[begin+1:end])
end

Base.length(traj::AbstractTrajectory) = length(traj.t)
Base.firstindex(traj::AbstractTrajectory) = 1
Base.lastindex(traj::AbstractTrajectory) = length(traj)

function duration(traj::AbstractTrajectory)
    if length(traj) == 0
        return 0.0
    end
    traj.t[end] - traj.t[begin]
end

SciMLBase.getsyms(traj::AbstractTrajectory) = traj.syms
SciMLBase.getindepsym(traj::AbstractTrajectory) = traj.indepsym

struct TrajectoryTuplesIter{T}
    traj::T
end

RecursiveArrayTools.tuples(traj::AbstractTrajectory) = TrajectoryTuplesIter(traj)

function Base.iterate(iter::TrajectoryTuplesIter, index=1)
    traj = iter.traj
    index > length(traj) && return nothing
    (traj.u[index], traj.t[index], traj.i === nothing ? 0 : traj.i[index]), index + 1
end

Base.eltype(::Type{TrajectoryTuplesIter{T}}) where {uType,tType,A,T<:AbstractTrajectory{uType,tType,A}} = Tuple{eltype(A),tType,Int}
Base.length(iter::TrajectoryTuplesIter) = length(iter.traj)

SciMLBase.getsyms(iter::TrajectoryTuplesIter) = iter.traj.syms
SciMLBase.getindepsym(iter::TrajectoryTuplesIter) = iter.traj.indepsym

"""
    collect_trajectory(iter, nocopy=false)

Create a `Trajectory` from an iterator that yields a sequence of 3-element tuples `(u, t, i)`.

If `nocopy` is true, the `u` vectors of the tuple will not be copied before adding them to the
trajectory.
"""
function collect_trajectory(iter; nocopy=false)
    (u, t, i), state = iterate(iter)

    indepsym = SciMLBase.getindepsym(iter)
    syms = SciMLBase.getsyms(iter)

    traj = Trajectory([nocopy ? u : copy(u)], [t], Int[i], syms=syms, indepsym=indepsym)

    for (u, t, i) in Iterators.rest(iter, state)
        push!(traj.t, t)
        push!(traj.u, nocopy ? u : copy(u))
        push!(traj.i, i)
    end

    traj
end

function Base.show(io::IO, m::MIME"text/plain", x::AbstractTrajectory)
    print(io, "t: ")
    show(io, m, x.t)
    println(io)
    print(io, "u: ")
    show(io, m, x.u)
    println(io)
    if x.i !== nothing
        print(io, "i:")
        show(io, m, x.i)
    end
end

@recipe function f(traj::AbstractTrajectory{U,T,A}) where {U,T,A}
    seriestype --> :steppre
    xguide --> ((traj.indepsym !== nothing) ? string(traj.indepsym) : "")
    label --> ((traj.syms !== nothing) ? reshape(string.(traj.syms), 1, :) : "")
    traj.t, traj'
end

struct Chain{V,T<:AbstractVector,U<:AbstractVector} <: AbstractVector{V}
    head::T
    tail::U

    function Chain(head::AbstractVector{X}, tail::AbstractVector{Y}) where {X,Y}
        new{promote_type(X, Y),typeof(head),typeof(tail)}(head, tail)
    end
end

Base.IndexStyle(::Type{<:Chain}) = IndexLinear()
Base.copy(ch::Chain) = Chain(copy(ch.head), copy(ch.tail))
Base.size(ch::Chain) = size(ch.head) .+ size(ch.tail)
@inline Base.firstindex(ch::Chain) = 1
@inline Base.lastindex(ch::Chain) = length(ch)
Base.@propagate_inbounds function Base.getindex(ch::Chain{V}, i::Int) where {V}
    if i > length(ch.head)
        val = ch.tail[i-length(ch.head)]
        convert(V, val)
    else
        val = ch.head[i]
        convert(V, val)
    end
end
Base.setindex!(ch::Chain, v, i::Int) = i > length(ch.head) ? ch.tail[i-length(ch.head)] = v : ch.head[i] = v


struct MergeTrajectory{T1,T2}
    traj1::T1
    traj2::T2
end

Base.IteratorSize(::Type{MergeTrajectory{T1,T2}}) where {T1,T2} = Base.SizeUnknown()
function Base.eltype(::Type{MergeTrajectory{T1,T2}}) where {T1,T2}
    E1 = eltype(T1)
    E2 = eltype(T2)
    U1 = fieldtypes(E1)[1]
    U2 = fieldtypes(E2)[1]
    T = fieldtypes(E1)[2]
    Tuple{Chain{promote_type(eltype(U1), eltype(U2)),U1,U2},T,Int}
end

function merge_next(((u1, t1, i1), state1)::Tuple, ((u2, t2, i2), state2)::Tuple)
    t, i = if t1 <= t2
        t1, i1
    else
        t2, i2
    end
    Chain(copy(u1), copy(u2)), t, i
end

function advance_next(mtraj::MergeTrajectory, ((u1, t1, i1), state1)::A, ((u2, t2, i2), state2)::B)::Union{Nothing,Tuple{A,B}} where {A<:Tuple,B<:Tuple}
    if t1 < t2
        next1 = iterate(mtraj.traj1, state1)
        if next1 === nothing
            return nothing
        end
        next1, ((u2, t2, i2), state2)
    elseif t1 > t2
        next2 = iterate(mtraj.traj2, state2)
        if next2 === nothing
            return nothing
        end
        ((u1, t1, i1), state1), next2
    elseif t1 == t2
        next1 = iterate(mtraj.traj1, state1)
        next2 = iterate(mtraj.traj2, state2)
        (next1 === nothing || next2 === nothing) && return nothing
        next1, next2
    else # t1 or t2 is NaN
        nothing
    end
end

function Base.iterate(mtraj::MergeTrajectory)
    s1 = iterate(mtraj.traj1)
    s2 = iterate(mtraj.traj2)
    if s1 === nothing || s2 === nothing
        return nothing
    end
    m = merge_next(s1, s2)
    (m, (s1, s2))
end

function Base.iterate(mtraj::MergeTrajectory, (s1, s2)::Tuple)
    s = advance_next(mtraj, s1, s2)
    if s === nothing
        return nothing
    end
    s1, s2 = s
    m = merge_next(s1, s2)
    (m, s)
end

SciMLBase.getsyms(iter::MergeTrajectory) = nothing
SciMLBase.getindepsym(iter::MergeTrajectory) = nothing

trajectory_iterator(traj::AbstractTrajectory) = tuples(traj)
trajectory_iterator(traj) = traj

# merge more than 2 trajectories using recursion
merge_trajectories(traj) = trajectory_iterator(traj)
function merge_trajectories(traj1, traj2)
    MergeTrajectory(merge_trajectories(traj1), merge_trajectories(traj2))
end
function merge_trajectories(traj1, traj2, other_trajs...)
    merge12 = merge_trajectories(traj1, traj2)
    merge_trajectories(merge12, other_trajs...)
end


function start_collect(::Nothing, indices, syms, indepsym)
    # return empty trajectory
    Trajectory(Vector{Int16}[], Float64[], Int[], syms=syms, indepsym=indepsym)
end

function start_collect(((u, t, i), state)::Tuple, indices, syms, indepsym)
    Trajectory([u[indices]], typeof(t)[], typeof(i)[], syms=syms, indepsym=indepsym)
end

function step_collect!(result::Trajectory, ::Nothing, (uprev, tprev, iprev)::Tuple, indices)
    # end trajectory
    push!(result.t, tprev)
    push!(result.i, iprev)
    result
end

function step_collect!(
    result::Trajectory,
    ((u, t, i), state)::Tuple,
    (uprev, tprev, iprev)::Tuple,
    indices
)
    if (@view u[indices]) != result.u[end]
        push!(result.u, u[indices])
        push!(result.t, tprev)
        push!(result.i, iprev)
    end
    result
end

function collect_sub_trajectory(iter, indices)
    indepsym = SciMLBase.getindepsym(iter)
    syms = SciMLBase.getsyms(iter)

    f = iterate(iter)
    traj = start_collect(f, indices, syms[indices], indepsym)
    while f !== nothing
        val, state = f
        f = iterate(iter, state)
        traj = step_collect!(traj, f, val, indices)
    end
    traj
end

function collect_sub_trajectories(iter, indices_list...)
    indepsym = SciMLBase.getindepsym(iter)
    syms = SciMLBase.getsyms(iter)

    f = iterate(iter)
    trajs = map(indices -> start_collect(f, indices, syms[indices], indepsym), indices_list)
    while f !== nothing
        val, state = f
        f = iterate(iter, state)
        foreach((traj, indices) -> step_collect!(traj, f, val, indices), trajs, indices_list)
    end
    trajs
end