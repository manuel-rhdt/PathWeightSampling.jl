using Random
import StatsBase

"""
An empirical distribution, defined by a histogram which assigns probabilities to
a set of axis-coordinates in a (possibly high-dimensional) grid.
"""
struct EmpiricalDistribution{Dims, T}
    prob::Array{Float64, Dims}
    axes::SVector{Dims, Vector{T}}
end

Base.eltype(::Type{<:EmpiricalDistribution{Dims, T}}) where {Dims, T} = Vector{T}

function empirical_dist(prob::AbstractArray{Float64,N}, axes::Vararg{<:AbstractVector,N}) where {N}
    EmpiricalDistribution{N, eltype(axes[1])}(prob ./ sum(prob), (axes...,))
end

function Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{<:EmpiricalDistribution})
    dist = d[]
	x = StatsBase.sample(rng, StatsBase.weights(vec(dist.prob)))
	index = CartesianIndices(size(dist.prob))
	i = index[x]
	map(j->dist.axes[j][i[j]], 1:length(i))
end

function Distributions.logpdf(dist::EmpiricalDistribution, v::AbstractVector)
	indices = [findfirst(==(v[i]), dist.axes[i]) for i=eachindex(dist.axes)]
	if nothing ∈ indices
		-Inf64
	else
		log(dist.prob[indices...])
	end
end
Distributions.logpdf(dist::EmpiricalDistribution, v) = logpdf(dist, [v])

function Distributions.pdf(dist::EmpiricalDistribution, v::AbstractVector)
	indices = [findfirst(==(v[i]), dist.axes[i]) for i=eachindex(dist.axes)]
	if nothing ∈ indices
		0.0
	else
		dist.prob[indices...]
	end
end
Distributions.pdf(dist::EmpiricalDistribution, v) = pdf(dist, [v])

function marginalize(dist::EmpiricalDistribution, axis_index::Integer)
	new_prob = sum(dist.prob, dims=axis_index)
	new_prob = dropdims(new_prob, dims=axis_index)
	new_axes = deleteat(dist.axes, axis_index)
	EmpiricalDistribution(new_prob, new_axes)
end

Base.getindex(dist::EmpiricalDistribution, axis_index::Integer) = marginalize(dist, axis_index)
Base.getindex(dist::EmpiricalDistribution, axis_index::AbstractVector{<:Integer}) = marginalize(dist, axis_index...)
