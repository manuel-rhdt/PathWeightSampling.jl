module GaussianMcmc

using LinearAlgebra
using Distributions

using Random
import Base.copy
import Base.copy!

include("fast_mvnormal.jl")

import Base.similar

function time_matrix(n::Integer, delta_t::Real)
    t_range = range(0.0, length=n, step=delta_t)
    transpose(t_range) .- t_range
end

# A collection of common parameters
struct System
    κ::Float64
    λ::Float64
    ρ::Float64
    μ::Float64
end

Base.broadcastable(system::System) = Ref(system)

function corr_ss(system::System)
    (t) -> system.κ / system.λ * exp(-abs(t) * system.λ)
end

function corr_xs(system::System)
    function (t)
        if t >= 0
            system.ρ * system.κ / system.λ / (system.λ + system.μ) * exp(-system.λ * t)
        else
            corr_sx(system)(-t)
        end
    end
end

function corr_sx(system::System)
    function (t)
        if t >= 0
            a = system.ρ * system.κ / system.λ / (system.λ - system.μ)
            b1 = (1 + (system.λ - system.μ) / (system.λ + system.μ)) * exp(-system.μ * t)
            b2 = -exp(-system.λ * t)
            a * (b1 + b2)
        else
            corr_xs(system)(-t)
        end
    end
end

function corr_xx(system::System)
    function (t)
        c1 = exp(-system.μ * abs(t)) -  exp(-system.λ * abs(t))
        c2 = exp(-system.μ * abs(t))
        d1 = system.ρ^2 / (system.λ^2 - system.μ^2) * system.κ / system.λ
        d2 = (1 + system.ρ / (system.λ + system.μ)) * system.κ / system.λ * system.ρ / system.μ
        d1 * c1 + d2 * c2
    end
end

function corr_z(system::System, t)
    [corr_ss(system).(t) corr_xs(system).(t); corr_sx(system).(t) corr_xx(system).(t)]
end

function prior(system::System, t)
    MvNormal(corr_ss(system).(t))
end

function marginal(system::System, t)
    MvNormal(corr_xx(system).(t))
end

function joint(system::System, t)
    MvNormal(corr_z(system, t))
end

function likelihood(system::System, t; signal::AbstractVector)
    c_ss = corr_ss(system).(t)
    c_sx = corr_sx(system).(t)
    c_xs = corr_xs(system).(t)
    c_xx = corr_xx(system).(t)
        
    regression_coef = c_sx * inv(c_ss)    
    p_x_given_s_cov = LinearAlgebra.Symmetric(c_xx - regression_coef * c_xs)
    
    MvNormal(regression_coef * signal, p_x_given_s_cov)
end

function posterior(system::System, t; response::AbstractVector)
    c_ss = corr_ss(system).(t)
    c_sx = corr_sx(system).(t)
    c_xs = corr_xs(system).(t)
    c_xx = corr_xx(system).(t)
        
    regression_coef = c_xs * inv(c_xx)    
    p_s_given_x_cov = LinearAlgebra.Symmetric(c_ss - regression_coef * c_sx)
    
    MvNormal(regression_coef * response, p_s_given_x_cov)
end

System() = System(0.25, 0.005, 0.01, 0.01)

function log_likelihood(system::System, t; signal::AbstractArray, response::AbstractArray)
    c_ss = corr_ss(system).(t)
    c_sx = corr_sx(system).(t)
    c_xs = corr_xs(system).(t)
    c_xx = corr_xx(system).(t)
        
    regression_coef = c_sx * inv(c_ss)    
    p_x_given_s_cov = LinearAlgebra.Symmetric(c_xx - regression_coef * c_xs)
    
    scaled_normal = MvNormal(p_x_given_s_cov)
    
    logpdf(scaled_normal, response .- regression_coef * signal)
end

function log_joint(system::System, t; signal::AbstractArray, response::AbstractArray)
    c_z = corr_z(system, t)
    distr = MvNormal(c_z)
    logpdf(distr, vcat(signal, response))
end

function log_prior(system::System, t; signal::AbstractArray)
    c_ss = corr_ss(system).(t)
    distr = MvNormal(c_ss)
    logpdf(distr, signal)
end

function energy(conf::Array{Float64,1}, joint)
    -logpdf(joint, conf)
end

function energy(conf::Array{Float64,1}, prior, joint, θ::Float64)    
    result = 0.0
    n_dim = length(conf) ÷ 2
    if θ > 0.0
        result += θ * logpdf(joint, conf)
    end
    if (1.0 - θ) > 0.0
        result += (1.0 - θ) * logpdf(prior, @view conf[1:n_dim])
    end
    -result
end

abstract type SystemConfiguration end

energy(conf::SystemConfiguration, joint) = energy(conf.state, joint)
energy(conf::SystemConfiguration, prior, joint, θ::Float64) = energy(conf.state, prior, joint, θ)

struct Mcmc{Conf <: SystemConfiguration,Prior,Joint}
    scale::Float64
    subsample::Int64
    prior::Prior
    joint::Joint
    theta::Float64
    initial::Conf
end

function Base.iterate(iter::Mcmc, state=iter.initial)
    current_energy = energy(state, iter.prior, iter.joint, iter.theta)
    
    accepted = 0
    rejected = 0
    
    current_conf = copy(state)
    new_conf = similar(state)
    
    while true
        propose_conf!(new_conf, current_conf, iter.scale)
        
        new_energy = energy(new_conf, iter.prior, iter.joint, iter.theta)
        
        if rand() < exp(current_energy - new_energy)
            accepted += 1
            current_energy = new_energy
            copy!(current_conf, new_conf)
        else
            rejected += 1
        end
        
        if (accepted + rejected) == iter.subsample + 1
            acceptance_rate = accepted / (rejected + accepted)
            return (current_conf, acceptance_rate), current_conf
        end
    end
end

copy(conf::T) where {T <: SystemConfiguration} = T(copy(conf.state))
copy!(dest::T, src::T) where {T <: SystemConfiguration} = copy!(dest.state, src.state)
similar(conf::T) where {T <: SystemConfiguration} = T(similar(conf.state))
struct UniformProposal{T} <: SystemConfiguration
    state::T
end

function propose_conf!(new_conf::UniformProposal{T}, previous_conf::UniformProposal{T}, scale::Float64) where {T <: AbstractArray{Float64,1}}
    rand!(new_conf.state)
    new_conf.state .-= 0.5
    new_conf.state .*= scale * 2.0
    new_conf.state .+= previous_conf.state
    nothing
end

struct GaussianProposal{T} <: SystemConfiguration
    state::T
end

function propose_conf!(new_conf::GaussianProposal{T}, previous_conf::GaussianProposal{T}, scale::Float64) where {T <: AbstractArray{Float64,1}}
    n_dim = length(previous_conf.state) ÷ 2
    new_conf.state .= previous_conf.state
    normal_distr = Normal(0, scale)
    for i in 1:n_dim
        new_conf.state[i] += rand(normal_distr)
    end
    nothing
end


function generate_mcmc_samples(initial::SystemConfiguration, num_samples::Integer, system::System, t::Matrix; scale::Real, subsample::Integer, θ::Real=1.0)
    prior_distr = FastMvNormal(corr_ss(system).(t))
    joint_distr = FastMvNormal(corr_z(system, t))
    
    mcmc = Mcmc(scale, subsample, prior_distr, joint_distr, θ, initial)

    samples = zeros((size(t, 1) * 2, num_samples))
    acceptance = zeros(num_samples)
    for (index, (sample, rate)) in Iterators.enumerate(Iterators.take(mcmc, num_samples))
        samples[:, index] = sample.state
        acceptance[index] = rate
    end

    samples, acceptance
end


mutable struct ThermodynamicIntegral
    system::System
    t::Matrix
    initial::SystemConfiguration
    scale::Real
    subsample::Integer
    samples::Array{Float64,3}
    acceptance_rates::Array{Float64,2}
    θ::Vector{Float64}
end

function ThermodynamicIntegral(system::System, t::Matrix, initial::SystemConfiguration, scale::Real, subsample::Integer)
    ThermodynamicIntegral(system, t, initial, scale, subsample, Array{Float64}(undef, (0, 0, 0)), Array{Float64}(undef, (0, 0)), Vector{Float64}(undef, 0))
end

function perform(integral::ThermodynamicIntegral, num_samples::Integer, θ::Real)
    perform(integral, num_samples, [θ])
end

function perform(integral::ThermodynamicIntegral, num_samples::Integer, θ::AbstractVector{<:Real})
    samples = []
    acceptance = []
    for θ in θ
        s, a = generate_mcmc_samples(integral.initial, num_samples, integral.system, integral.t, scale=integral.scale, subsample=integral.subsample, θ=θ)
        push!(samples, s)
        push!(acceptance, a)
    end
    integral.samples = cat(samples..., dims=3)
    integral.acceptance_rates = cat(acceptance..., dims=2)
    integral.θ = collect(θ)
    nothing
end

function potential(integral::ThermodynamicIntegral)
    n_dim = size(integral.t, 1)
    signal = @view integral.samples[1:n_dim,:,:]
    response = @view integral.samples[n_dim + 1:end, 1,:]

    ll = map(integral.θ, eachslice(signal, dims=3), eachcol(response)) do θ, sig, res
        log_likelihood(integral.system, integral.t, signal=sig, response=res)
    end

    hcat(ll...)
end


function estimate_marginal_density(initial::SystemConfiguration, num_samples::Integer, system::System, t::Matrix; scale::Real, subsample::Integer, θ::Real=1.0)
    samples, acceptance = generate_mcmc_samples(initial, num_samples, system, t, scale=scale, subsample=subsample, θ=θ)
    n_dim = Int(size(t, 1))
    signal = @view samples[1:n_dim,:]
    response = @view initial.state[n_dim + 1:end]
    log_likelihood(system, t, signal=signal, response=response), acceptance
end


export System,
    corr_ss, corr_sx, corr_xs, corr_xx, corr_z,
    prior, marginal, joint, likelihood, posterior, log_likelihood, log_joint, log_prior,
    estimate_marginal_density, generate_mcmc_samples,
    time_matrix,
    GaussianProposal, UniformProposal,
    ThermodynamicIntegral, perform, potential,
    WangLandau, flatness

include("wang_landau.jl")
include("comparison.jl")

module Trajectories

using StaticArrays

include("trajectory_moves.jl")
include("thermodynamic_integral.jl")

export Trajectory, TIEstimate, AnnealingEstimate, log_marginal

end

end # module
