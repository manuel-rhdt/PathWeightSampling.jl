using RecipesBase
using DiffEqBase
import StaticArrays:SVector

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

Base.copy(traj::Trajectory) = Trajectory(copy(traj.t), deepcopy(traj.u), copy(traj.i))

Base.getindex(traj::Trajectory, idx::Int) = (traj.u[idx], traj.t[idx], idx > length(traj.i) ? 0 : traj.i[idx])
Base.:(==)(traj1::Trajectory, traj2::Trajectory) = (traj1.t == traj2.t) && (traj1.u == traj2.u) && (traj1.i == traj2.i)

function (t::Trajectory)(time::Real)
    index = searchsortedfirst(t.t, time)
    if t.t[index] == time
        index += 1
    end
    if index > lastindex(t.t)
        error("Can't access trajectory that ends at t=$(last(t.t)) at time $time.")
    end
    t.u[index]
end

function Trajectory(u::AbstractMatrix{T}, t::AbstractVector{tType}, i=Int[]) where {tType <: Real,T <: Real}
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

function Base.iterate(traj::Trajectory, index=1)
    index > length(traj) && return nothing
    traj[index], index + 1
end

Base.eltype(::Type{T}) where {uType,tType,T <: AbstractTrajectory{uType,tType}} = Tuple{uType,tType,Int}

function collect_trajectory(iter, nocopy=false)
    ((u, t, i), state) = iterate(iter)
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
