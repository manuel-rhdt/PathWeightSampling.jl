using RecipesBase
using DiffEqBase
import StaticArrays:SVector

abstract type AbstractTrajectory{uType,tType} end

struct Trajectory{uType,tType} <: AbstractTrajectory{uType,tType}
    t::Vector{tType} # vector of length N, recording the reaction times and the start/end times of the trajectory
    u::Vector{uType} # vector of length N, recording the copy numbers at the corresponding time
    i::Vector{Int} # vector of length N-2 recording all reaction indices, or empty vector if no reaction indices
end

Trajectory(t::Vector, u::Vector) = Trajectory(t, u, Int[])

Base.copy(traj::Trajectory) = Trajectory(copy(traj.t), copy(traj.u), copy(traj.i))

Base.getindex(traj::Trajectory, i::Int) = traj.u[i]
Base.getindex(traj::Trajectory, i::AbstractRange) = traj.u[i]
Base.:(==)(traj1::Trajectory, traj2::Trajectory) = (traj1.t == traj2.t) && (traj1.u == traj2.u)

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

mutable struct PartialTrajectory{uType,tType,N,NPart} <: AbstractTrajectory{SVector{NPart,uType},tType}
    idxs::SVector{NPart,Int}
    t::Vector{tType}
    u::Vector{SVector{N,uType}}
end

Base.copy(traj::PartialTrajectory) = PartialTrajectory(traj.idxs, copy(traj.t), copy(traj.u))
Base.getindex(traj::PartialTrajectory, i::Int) = traj.u[i][traj.idxs]
Base.getindex(traj::PartialTrajectory, i::AbstractRange) = [v[traj.idxs] for v in traj.u[i]]

Trajectory(partial::PartialTrajectory) = convert(Trajectory, partial)

function Base.convert(::Type{Trajectory}, partial::PartialTrajectory{uType,tType,M,N}) where {uType,tType,M,N}
    t = copy(partial.t)
    u = partial[begin:end]

    i = 2
    while i < length(u)
        du = u[i] - u[i - 1]
        if all(du .== 0)
            popat!(u, i)
            popat!(t, i)
        else
            i += 1
        end
    end

    Trajectory(t, u)
end

function duration(traj::AbstractTrajectory)
    if length(traj) == 0
        return 0.0
    end
    traj.t[end] - traj.t[begin]
end

function trajectory(sol::ODESolution{T,N}, idxs) where {T,N}
    idxs = SVector{length(idxs),Int}(idxs)
    PartialTrajectory(idxs, sol.t, get_u(sol))
end

Base.length(traj::AbstractTrajectory) = length(traj.u)
Base.firstindex(traj::AbstractTrajectory) = 1
Base.lastindex(traj::AbstractTrajectory) = length(traj)

function Base.iterate(traj::Trajectory, index=1)
    if index > length(traj)
        return nothing
    end

    ridx = checkbounds(Bool, traj.i, index - 1) ? traj.i[index - 1] : 0

    (traj.u[index], traj.t[index], ridx), index + 1
end

function Base.iterate(traj::PartialTrajectory, index=1)
    if index > length(traj)
        return nothing
    end

    (traj.u[index][traj.idxs], traj.t[index]), index + 1
end

Base.eltype(::Type{T}) where {uType,tType,T <: AbstractTrajectory{uType,tType}} = Tuple{uType,tType,Int}

struct MergeTrajectory{uType,tType,T1 <: Trajectory{uType,tType},T2 <: Trajectory{uType,tType}}
    first::T1
    second::T2
end

Base.merge(traj1::T1, traj2::T2) where {uType,tType,T1 <: Trajectory{uType,tType},T2 <: Trajectory{uType,tType}} = MergeTrajectory{uType,tType,T1,T2}(traj1, traj2)
Base.IteratorSize(::Type{MergeTrajectory{uType,tType,T1,T2}}) where {uType,tType,T1,T2} = Base.SizeUnknown()

function duration(traj::MergeTrajectory)
    max(traj.first.t[end], traj.second.t[end]) - min(traj.first.t[begin], traj.second.t[begin])
end

function Base.iterate(iter::MergeTrajectory)
    if length(iter.first) == 0 || length(iter.second) == 0
        return nothing
    end
    iterate(iter, (1, 1, min(iter.first.t[begin], iter.second.t[begin]), vcat(iter.first.u[begin], iter.second.u[begin])))
end

function Base.iterate(iter::MergeTrajectory{uType,tType}, (i, j, t, u)::Tuple{Int,Int,tType,<:AbstractVector{uType}}) where {uType,tType}
    current_t = t

    if t == Inf
        return nothing
    end

    if (i + 1) > length(iter.first)
        t_i = Inf
    else 
        @inbounds t_i = iter.first.t[i + 1]
    end

    if (j + 1) > length(iter.second)
        t_j = Inf
    else
        @inbounds t_j = iter.second.t[j + 1]
    end

    if t_i == t_j == Inf
        return (u, current_t), (i, j, Inf, u)
    end

    next_u = u

    if t_i <= t_j
        t = t_i
        i = i + 1
        u_i = iter.first.u[i]
        for (i, x) in enumerate(u_i)
            next_u = setindex(next_u, x, i)
        end
    else
        t = t_j
        j = j + 1
        u_j = iter.second.u[j]
        for (i, x) in enumerate(u_j)
            next_u = setindex(next_u, x, length(next_u) - length(u_j) + i)
        end
    end

    (u, current_t), (i, j, t, next_u)
end

@recipe function f(traj::AbstractTrajectory{uType,tType}) where {uType,tType}
    seriestype --> :steppost
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
