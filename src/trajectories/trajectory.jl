using RecipesBase
using DiffEqBase
import StaticArrays: SVector

abstract type AbstractTrajectory{uType,tType} end

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
struct Trajectory{uType<:AbstractVector,tType<:Real} <: AbstractTrajectory{uType,tType}
    u::Vector{uType} # vector of length N. u[i] is the vector of copy numbers in the interval from t[i-1] to t[i]
    t::Vector{tType} # vector of length N, recording the reaction times and the end time of the trajectory
    i::Vector{Int} # vector of length N recording all reaction indices, or empty vector if no reaction indices
end

Trajectory(u::Vector{<:AbstractVector}, t::Vector) = Trajectory(u, t, Int[])

Base.copy(traj::Trajectory) = Trajectory(copy(traj.u), deepcopy(traj.t), copy(traj.i))

Base.getindex(traj::Trajectory, idx::Int) = (traj.u[idx], traj.t[idx], idx > length(traj.i) ? 0 : traj.i[idx])
Base.:(==)(traj1::Trajectory, traj2::Trajectory) = (traj1.t == traj2.t) && (traj1.u == traj2.u) && (traj1.i == traj2.i)

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

function Trajectory(u::AbstractMatrix{T}, t::AbstractVector{tType}, i = Int[]) where {tType<:Real,T<:Real}
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

function Base.iterate(traj::Trajectory, index = 1)
    index > length(traj) && return nothing
    traj[index], index + 1
end

Base.eltype(::Type{T}) where {uType,tType,T<:AbstractTrajectory{uType,tType}} = Tuple{uType,tType,Int}

"""
    collect_trajectory(iter, nocopy=false)

Create a `Trajectory` from an iterator that yields a sequence of 3-element tuples `(u, t, i)`.

If `nocopy` is true, the `u` vectors of the tuple will not be copied before adding them to the
trajectory.
"""
function collect_trajectory(iter; nocopy = false)
    (u, t, i), state = iterate(iter)
    traj = Trajectory([nocopy ? u : copy(u)], [t], Int[i])

    for (u, t, i) in Iterators.rest(iter, state)
        push!(traj.t, t)
        push!(traj.u, nocopy ? u : copy(u))
        push!(traj.i, i)
    end

    traj
end

@recipe function f(traj::AbstractTrajectory{uType,tType}) where {uType,tType}
    seriestype --> :steppre
    # label --> hcat([String(sym) for sym in traj.syms]...)

    N = length(traj.u[1])
    uvec = zeros(eltype(uType), length(traj), N)
    tvec = zeros(tType, length(traj))
    for (i, (u, t)) in enumerate(traj)
        uvec[i, :] = u
        tvec[i] = t
    end

    tvec, uvec
end

struct MergeTrajectory{T1,T2}
    traj1::T1
    traj2::T2
end

Base.IteratorSize(::Type{MergeTrajectory{T1,T2}}) where {T1,T2} = Base.SizeUnknown()
function Base.eltype(::Type{MergeTrajectory{T1,T2}}) where
{T,U1,U2,T1<:AbstractTrajectory{U1,T},T2<:AbstractTrajectory{U2,T}}
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

function advance_next(mtraj::MergeTrajectory, ((u1, t1, i1), state1)::A, ((u2, t2, i2), state2)::B)::Union{Nothing, Tuple{A, B}} where {A<:Tuple,B<:Tuple}
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
    if s1 === nothing || s2 === nothing return nothing end
    m = merge_next(s1, s2)
    (m, (s1, s2))
end

function Base.iterate(mtraj::MergeTrajectory, (s1, s2)::Tuple)
    s = advance_next(mtraj, s1, s2)
    if s === nothing return nothing end
    s1, s2 = s
    m = merge_next(s1, s2)
    (m, s)
end