using DifferentialEquations

abstract type AbstractTrajectory{uType,tType,N} end

mutable struct Trajectory{uType,tType,N} <: AbstractTrajectory{uType,tType,N}
    syms::SVector{N, Symbol}
    t::Vector{tType}
    u::Vector{SVector{N, uType}}
end

Base.copy(traj::Trajectory) = Trajectory(copy(traj.syms), copy(traj.t), copy(traj.u))
Base.getindex(traj::Trajectory, i::Int) = traj.u[i]
Base.getindex(traj::Trajectory, i::AbstractRange) = traj.u[i]
Base.:(==)(traj1::Trajectory, traj2::Trajectory) = (traj1.syms == traj2.syms) && (traj1.t==traj2.t) && (traj1.u==traj2.u)

function Base.copyto!(to::Trajectory, from::Trajectory)
    resize!(to.t, length(from.t))
    resize!(to.u, length(from.u))
    copyto!(to.t, from.t)
    copyto!(to.u, from.u)
    to
end

function trajectory(sol::ODESolution{T,N,Vector{SVector{M, T}}}) where {T, N, M}
    variables = species(sol.prob.f.f)
    symbols = SVector{M}([v.name for v in variables]::Vector{Symbol})
    Trajectory(symbols, sol.t, sol.u)
end

mutable struct PartialTrajectory{uType,tType,N,NPart} <: AbstractTrajectory{uType,tType,NPart}
    syms::SVector{NPart, Symbol}
    idxs::SVector{NPart, Int}
    t::Vector{tType}
    u::Vector{SVector{N, uType}}
end

Base.copy(traj::PartialTrajectory) = PartialTrajectory(traj.syms,traj.idxs, copy(traj.t), copy(traj.u))
Base.getindex(traj::PartialTrajectory, i::Int) = traj.u[i][traj.idxs]
Base.getindex(traj::PartialTrajectory, i::AbstractRange) = [v[traj.idxs] for v in traj.u[i]]

function Base.convert(::Type{Trajectory}, partial::PartialTrajectory{uType,tType,M,N}) where {uType,tType,M,N}
    Trajectory(partial.syms, partial.t, partial[begin:end])
end

function trajectory(sol::ODESolution{T,N,Vector{SVector{M, T}}}, syms::SVector{NPart, Symbol}) where {T, N, M, NPart}
    idxs = Int[]

    variables = species(sol.prob.f.f)
    symbols = SVector{M}([v.name for v in variables]::Vector{Symbol})
    for (j, sym) in enumerate(syms)
        i = findfirst(isequal(sym), symbols)
        push!(idxs, i)
    end

    idxs = SVector{NPart, Int}(idxs)
    PartialTrajectory(syms, idxs, sol.t, sol.u)
end

function trajectory(sol::ODESolution{T,N,Vector{SVector{M, T}}}, syms::SVector{NPart, Symbol}, idxs::SVector{NPart, Int}) where {T, N, M, NPart}
    PartialTrajectory(syms, idxs, sol.t, sol.u)
end

Base.length(traj::AbstractTrajectory) = length(traj.u)
Base.firstindex(traj::AbstractTrajectory) = 1
Base.lastindex(traj::AbstractTrajectory) = length(traj)

function Base.iterate(traj::Trajectory, index=1)
    if index > length(traj)
        return nothing
    end

    (traj.u[index], traj.t[index]), index + 1
end

function Base.iterate(traj::PartialTrajectory, index=1)
    if index > length(traj)
        return nothing
    end

    (traj.u[index][traj.idxs], traj.t[index]), index + 1
end

Base.eltype(::Type{T}) where {uType,tType,N,T<:AbstractTrajectory{uType,tType,N}} = Tuple{SVector{N,uType}, tType}

struct MergeTrajectory{uType,tType,N,N1,N2,T1<:AbstractTrajectory{uType,tType,N1},T2<:AbstractTrajectory{uType,tType,N2}} <: AbstractTrajectory{uType,tType,N}
    syms::SVector{N, Symbol}
    first::T1
    second::T2
end

Base.merge(traj1::AbstractTrajectory, traj2::AbstractTrajectory) = MergeTrajectory(vcat(traj1.syms, traj2.syms), traj1, traj2)
Base.IteratorSize(::Type{MergeTrajectory{uType,tType,N,N1,N2,T1,T2}}) where {uType,tType,N,N1,N2,T1,T2} = Base.SizeUnknown()
Base.iterate(iter::MergeTrajectory{uType,tType,N}) where {uType, tType, N} = iterate(iter, (1, 1, min(iter.first.t[begin], iter.second.t[begin])))

function Base.iterate(iter::MergeTrajectory{uType,tType,N,N1,N2}, (i, j, t)::Tuple{Int, Int, tType}) where {uType, tType, N, N1, N2}
    if i > length(iter.first) && j > length(iter.second)
        return nothing
    end

    current_t = t

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

    u = @inbounds vcat(iter.first[min(i, length(iter.first))], iter.second[min(j, length(iter.second))])

    if t_i < t_j
        t = t_i
        i = i + 1
    elseif t_j < t_i
        t = t_j
        j = j + 1
    else
        t = t_i
        i = i + 1
        j = j + 1
    end

    (u, current_t), (i, j, t)
end

@recipe function f(traj::AbstractTrajectory{uType,tType,N}) where {uType,tType,N}
    seriestype --> :steppost
    label --> hcat([String(sym) for sym in traj.syms]...)

    uvec = zeros(uType, length(traj), N)
    tvec = zeros(tType, length(traj))
    for (i, (u, t)) in enumerate(traj)
        uvec[i, :] = u
        tvec[i] = t
    end

    tvec, uvec
end
