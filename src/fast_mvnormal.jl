using LinearAlgebra
import Distributions: logpdf

struct FastMvNormal
    e_val::Vector{Float64}
    prec_U::Matrix{Float64}
    logdetcov::Float64
end

function FastMvNormal(cov::Matrix{Float64})
    vals, vecs = eigen(cov)
    prec_U = vecs * Diagonal(sqrt.(1.0 ./ vals))
    logdetcov = log(2π) * length(vals) + sum(log.(vals))
    FastMvNormal(vals, prec_U, logdetcov)
end

function logpdf(acc::FastMvNormal, vec::AbstractVector{Float64})
    if length(vec) != size(acc.prec_U, 1)
        throw(DimensionMismatch("array wrong dimensions"))
    end
    
    prec_U = acc.prec_U
    
    maha = 0.0::Float64
    for j = axes(prec_U, 2)
        tmp = 0.0::Float64
        @simd for i = axes(prec_U, 1)
            @inbounds tmp += (vec[i] * prec_U[i, j])
        end
        maha += tmp^2
    end
    - 0.5 * (acc.logdetcov + maha)
end

function logpdf(acc::FastMvNormal, mat::AbstractMatrix{Float64})
    map(x->logpdf(acc, x), eachcol(mat))
end