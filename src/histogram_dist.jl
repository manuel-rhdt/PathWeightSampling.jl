using Distributions
import StatsBase
import Random: AbstractRNG

struct HistogramDistribution{T <: Real} <: DiscreteUnivariateDistribution
    weights::StatsBase.FrequencyWeights{Float64,Float64,Vector{Float64}}
    values::Vector{T}
    dict::Dict{T,Float64}
end

Base.rand(rng::AbstractRNG, d::HistogramDistribution) = StatsBase.sample(rng, d.values, d.weights)
Distributions.logpdf(d::HistogramDistribution, x::Real) = log(get(d.dict, x, 0.0))
Distributions.pdf(d::HistogramDistribution, x::Real) = get(d.dict, x, 0.0)
Distributions.logpdf(d::HistogramDistribution, x::Integer) = log(get(d.dict, x, 0.0))
Distributions.pdf(d::HistogramDistribution, x::Integer) = get(d.dict, x, 0.0)

struct MvHistogramDistribution{T <: AbstractVector} <: DiscreteMultivariateDistribution
    weights::StatsBase.FrequencyWeights{Float64,Float64,Vector{Float64}}
    values::Vector{T}
    dict::Dict{T,Float64}
end

function histogram_dist(values::AbstractVector{<:Real}, weights)
    mapping = StatsBase.proportionmap(values, StatsBase.fweights(weights))
    list = sort(collect(mapping), by=x -> x[2], rev=true)
    w = StatsBase.fweights(getfield.(list, 2))
    vals = getfield.(list, 1)
    HistogramDistribution(w, vals, mapping)
end

function mv_histogram_dist(values::AbstractArray, weights)
    mapping = StatsBase.proportionmap(values, StatsBase.fweights(weights))
    list = sort(collect(mapping), by=x -> x[2], rev=true)
    w = StatsBase.fweights(getfield.(list, 2))
    vals = getfield.(list, 1)
    MvHistogramDistribution(w, vals, mapping)
end

Base.length(d::MvHistogramDistribution) = length(get(d.values, 1, []))
Base.eltype(::MvHistogramDistribution{<:AbstractVector{T}}) where T = T

Distributions._rand!(rng::AbstractRNG, dist::MvHistogramDistribution{<:AbstractVector{T}}, x::AbstractArray{T,1}) where T = x[:] = StatsBase.sample(rng, dist.values, dist.weights)
Distributions._pdf(dist::MvHistogramDistribution{<:AbstractVector{T}}, val::AbstractVector{T}) where T = get(dist.dict, val, 0.0)
Distributions._logpdf(dist::MvHistogramDistribution{<:AbstractVector{T}}, val::AbstractVector{T}) where T = log(get(dist.dict, val, 0.0))

function StatsBase.entropy(d::Union{HistogramDistribution,MvHistogramDistribution})
    -sum(d.weights.values) do w
        p = w / d.weights.sum
        p * log(p)
    end
end
