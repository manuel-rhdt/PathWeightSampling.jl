using RecipesBase
using DiffEqBase
import StaticArrays:SVector

abstract type AbstractTrajectory{uType,tType} end

struct Trajectory{uType<:AbstractVector,tType<:Real} <: AbstractTrajectory{uType,tType}
    t::Vector{tType} # vector of length N, recording the reaction times and the end time of the trajectory
    u::Vector{uType} # vector of length N. u[i] records the copy numbers in the interval from t[i-1] to t[i]
    i::Vector{Int} # vector of length N recording all reaction indices, or empty vector if no reaction indices
end

Trajectory(t::Vector, u::Vector) = Trajectory(t, u, Int[])

Base.copy(traj::Trajectory) = Trajectory(copy(traj.t), copy(traj.u), copy(traj.i))

Base.getindex(traj::Trajectory, i::Int) = traj.u[i]
Base.getindex(traj::Trajectory, i::AbstractRange) = traj.u[i]
Base.:(==)(traj1::Trajectory, traj2::Trajectory) = (traj1.t == traj2.t) && (traj1.u == traj2.u) && (traj1.i == traj2.i)

function Base.copyto!(to::Trajectory, from::Trajectory)
    resize!(to.t, length(from.t))
    resize!(to.u, length(from.u))
    resize!(to.i, length(from.i))
    copyto!(to.t, from.t)
    copyto!(to.u, from.u)
    copyto!(to.i, from.i)
    to
end

function (t::Trajectory)(time::Real)
    index = clamp(searchsortedfirst(t.t, time), 2, length(t) + 1)
    t.u[index - 1]
end

function (t::Trajectory)(times::AbstractArray{<:Real})
    j = 2
    map(times) do time
        val = nothing
        while j <= length(t.t)
            if t.t[j] < time
                j += 1
                continue
            else
                val = t.u[j - 1]
                break
            end
        end
        val
    end
end

function clip!(t::Trajectory, time::Real)
    index = searchsortedfirst(t.t, time)
    resize!(t.t, index)
    resize!(t.u, index)
    t.t[index] = time
    t
end

function get_slice(t::Trajectory, tspan::Tuple{<:Real,<:Real})
    i1 = searchsortedfirst(t.t, tspan[1])
    i2 = searchsortedfirst(t.t, tspan[2])

    new_t = similar(t.t, i2 - i1 + 3)
    new_t[begin] = tspan[1]
    new_t[end] = tspan[2]
    new_t[begin + 1:end - 1] .= @view t.t[i1:i2]

    new_u = similar(t.u, i2 - i1 + 3)
    new_u[begin] = t.u[i1]
    new_u[end] = t.u[i2]
    new_u[begin + 1:end - 1] .= @view t.u[i1:i2]

    Trajectory(new_t, new_u)
end

function get_u(sol::ODESolution{T,N,Vector{SVector{M,T}}}) where {T,N,M}
    sol.u
end

function get_u(sol::ODESolution{T,N,Vector{Vector{T}}}) where {T,N}
    sol.u
end

Trajectory(sol::ODESolution) = trajectory(sol)

function Trajectory(t::AbstractVector{tType}, u::AbstractMatrix{T}, i=Int[]) where {tType <: Real,T <: Real}
    num_components = size(u, 1)
    if num_components > 0
        u_vec = [SVector{num_components}(c) for c in eachcol(u)]
    else
        u_vec = SVector{0,T}[]
    end

    Trajectory(convert(Vector{tType}, t), u_vec, i)
end

function trajectory(sol::ODESolution{T,N}) where {T,N}
    Trajectory(sol.t, get_u(sol))
end

function duration(traj::AbstractTrajectory)
    if length(traj) == 0
        return 0.0
    end
    traj.t[end] - traj.t[begin]
end

Base.length(traj::AbstractTrajectory) = length(traj.u)
Base.firstindex(traj::AbstractTrajectory) = 1
Base.lastindex(traj::AbstractTrajectory) = length(traj)

function Base.iterate(traj::Trajectory, index=1)
    if index > length(traj)
        return nothing
    end

    ridx = checkbounds(Bool, traj.i, index) ? traj.i[index] : 0

    (traj.u[index], traj.t[index], ridx), index + 1
end

Base.eltype(::Type{T}) where {uType,tType,T <: AbstractTrajectory{uType,tType}} = Tuple{uType,tType,Int}

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
