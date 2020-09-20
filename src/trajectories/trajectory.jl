using DifferentialEquations

mutable struct Trajectory{uType,tType} <: AbstractArray{uType, 2}
    syms::Vector{Symbol}
    t::Vector{tType}
    u::Array{uType,2}
end

Base.copy(traj::Trajectory) = Trajectory(copy(traj.syms), copy(traj.t), copy(traj.u))

function trajectory(sol::ODESolution)
    Trajectory(sol.prob.f.syms, sol.t, sol[:, :])
end

function trajectory(sol::ODESolution, syms::Vector{Symbol})
    idxs = UInt64[]
    for sym in syms
        i = findfirst(isequal(sym), sol.prob.f.syms)
        push!(idxs, i)
    end
    Trajectory(syms, sol.t, sol[idxs, :])
end

Base.size(traj::Trajectory) = size(traj.u)

Base.IndexStyle(::Trajectory) = IndexLinear()

Base.getindex(traj::Trajectory, i::Int) = getindex(traj.u, i)

struct LockstepIter{uType,tType}
    first::Trajectory{uType,tType}
    second::Trajectory{uType,tType}
end

Base.iterate(iter::LockstepIter{uType}) where uType = iterate(iter, (1, 1, min(iter.first.t[begin], iter.second.t[begin]), uType[]))

function Base.iterate(iter::LockstepIter{uType}, (i, j, t, u)::Tuple{Int64, Int64, Float64, Vector{uType}}) where uType
    if i > size(iter.first.t, 1) && j > size(iter.second.t, 1)
        return nothing
    end

    current_t = t

    if (i + 1) > size(iter.first.t, 1)
        t_i = Inf
    else 
        t_i = iter.first.t[i + 1]
    end

    if (j + 1) > size(iter.second.t, 1)
        t_j = Inf
    else
        t_j = iter.second.t[j + 1]
    end

    n1 = size(iter.first.u, 1)
    n2 = size(iter.second.u, 1)
    resize!(u, n1 + n2)
    u[begin:n1] = @view iter.first.u[:, i]
    u[n1+1:end] = @view iter.second.u[:, j]

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

    (u, current_t), (i, j, t, u)
end

@recipe function f(traj::Trajectory)
    seriestype --> :steppost
    label --> hcat([String(sym) for sym in traj.syms]...)

    plotvecs = []
    for i in 1:size(traj.u, 1)
        push!(plotvecs, traj.t)
        push!(plotvecs, traj.u[i, :])
    end

    traj.t, transpose(traj.u)
end
