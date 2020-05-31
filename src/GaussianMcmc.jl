module GaussianMcmc

using LinearAlgebra
using Distributions
using Random

include("fast_mvnormal.jl")

function time_matrix(n::Integer, delta_t::Real)
    t_range = range(0.0, length = n, step = delta_t)
    transpose(t_range) .- t_range
end

# A collection of common parameters
struct System
    λ::Float64
    κ::Float64
    ρ::Float64
    μ::Float64
end

function corr_ss(system::System)
    (t)->system.κ / system.λ * exp(-abs(t) * system.λ)
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

System() = System(0.005, 0.25, 0.01, 0.01)

function log_likelihood(system::System, t, signal::AbstractArray{T}, response::AbstractArray{T}) where {T <: Real}
    c_ss = corr_ss(system).(t)
    c_sx = corr_sx(system).(t)
    c_xs = corr_xs(system).(t)
    c_xx = corr_xx(system).(t)
        
    regression_coef = c_sx * inv(c_ss)    
    p_x_given_s_cov = LinearAlgebra.Symmetric(c_xx - regression_coef * c_xs)
    
    scaled_normal = MvNormal(p_x_given_s_cov)
    
    logpdf(scaled_normal, response .- regression_coef * signal)
end


function potential(conf::Array{Float64,1}, prior, joint, θ::Float64)    
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

function propose_conf!(new_conf::T, previous_conf::T, scale::Float64) where {T <: AbstractArray{Float64,1}}
    rand!(new_conf)
    new_conf .-= 0.5
    new_conf .*= scale
    new_conf .+= previous_conf
    nothing
end

struct Mcmc{Prior,Joint}
    scale::Float64
    skip::Int64
    prior::Prior
    joint::Joint
    theta::Float64
    initial::Array{Float64,1}
end

function Base.iterate(iter::Mcmc, state = iter.initial)
    current_pot = potential(state, iter.prior, iter.joint, iter.theta)
    
    accepted = 0
    rejected = 0
    
    new_conf = similar(state)
    
    while true
        propose_conf!(new_conf, state, iter.scale)
        
        new_pot = potential(new_conf, iter.prior, iter.joint, iter.theta)
        
        if rand() < exp(current_pot - new_pot)
            accepted += 1
            current_pot = new_pot
            acceptance_rate = accepted / rejected
            if accepted == iter.skip + 1
                return (new_conf, acceptance_rate), new_conf
            end
        else
            rejected += 1
        end
        
        if ((accepted + rejected) % (iter.skip * 100)) == 0
            acceptance_rate = accepted / rejected
            println("Slow convergence: acceptance rate $acceptance_rate = $accepted/$rejected")
        end
    end
end


function estimate_marginal_density(initial::AbstractArray{<:Real,1}, num_samples::Integer, system::System, t::Matrix; scale::Real, skip::Integer, θ::Real = 1.0)
    samples = zeros((length(initial), num_samples))
    acceptance = zeros(num_samples)
    
    prior_distr = FastMvNormal(corr_ss(system).(t))
    joint_distr = FastMvNormal(corr_z(system, t))
    
    mcmc = Mcmc(scale, skip, prior_distr, joint_distr, θ, copy(initial))
    
    for (index, (sample, rate)) in Iterators.enumerate(Iterators.take(mcmc, num_samples))
        samples[:, index] = sample
        acceptance[index] = rate
    end
    
    n_dim = Int(length(initial) // 2)
    signal = @view samples[1:n_dim,:]
    response = @view initial[n_dim + 1:end]
    log_likelihood(system, t, signal, response), acceptance
end


export System,
    corr_ss, corr_sx, corr_xs, corr_xx, corr_z,
    prior, marginal, joint, log_likelihood,
    estimate_marginal_density

end # module
