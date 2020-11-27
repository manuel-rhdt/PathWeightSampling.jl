using LinearAlgebra
using Distributions

function corr_ss(κ, λ, ρ, μ)
    (t) -> κ / λ * exp(-abs(t) * λ)
end

function corr_xs(κ, λ, ρ, μ)
    function (t)
        if t >= 0
            ρ * κ / λ / (λ + μ) * exp(-λ * t)
        else
            corr_sx(κ, λ, ρ, μ)(-t)
        end
    end
end

function corr_sx(κ, λ, ρ, μ)
    function (t)
        if t >= 0
            a = ρ * κ / λ / (λ - μ)
            b1 = (1 + (λ - μ) / (λ + μ)) * exp(-μ * t)
            b2 = -exp(-λ * t)
            a * (b1 + b2)
        else
            corr_xs(κ, λ, ρ, μ)(-t)
        end
    end
end

function corr_xx(κ, λ, ρ, μ)
    function (t)
        c1 = exp(-μ * abs(t)) -  exp(-λ * abs(t))
        c2 = exp(-μ * abs(t))
        d1 = ρ^2 / (λ^2 - μ^2) * κ / λ
        d2 = (1 + ρ / (λ + μ)) * κ / λ * ρ / μ
        d1 * c1 + d2 * c2
    end
end


struct GaussianSystem{Signal <: MvNormal,Joint <: MvNormal,Likelihood <: MvNormal}
    prior::Signal
    joint::Joint
    likelihood::Likelihood
end

function GaussianSystem(
    κ::Real, 
    λ::Real, 
    ρ::Real, 
    μ::Real; 
    duration::Real=1.0, 
    delta_t::Real=1.0, 
    t::Matrix{<:Real}=time_matrix(duration=duration, delta_t=delta_t)
)
    c_ss = corr_ss(κ, λ, ρ, μ).(t)
    c_sx = corr_sx(κ, λ, ρ, μ).(t)
    c_xs = corr_xs(κ, λ, ρ, μ).(t)
    c_xx = corr_xx(κ, λ, ρ, μ).(t)
    c_z = [c_ss c_xs; c_sx c_xx]

    prior = MvNormal(LinearAlgebra.Symmetric(c_ss))
    joint = MvNormal(LinearAlgebra.Symmetric(c_z))

    regression_coef = c_sx * inv(c_ss)    
    p_x_given_s_cov = LinearAlgebra.Symmetric(c_xx - regression_coef * c_xs)
    likelihood = MvNormal(p_x_given_s_cov)

    GaussianSystem(prior, joint, likelihood)
end

function GaussianSystem(;
    mean_s::Real=50, 
    tau_s::Real=1.0, 
    corr_time_ratio::Real=10, 
    mean_x::Real=mean_s, 
    tau_x::Real=tau_s / corr_time_ratio, 
    duration::Real=1.0, 
    delta_t::Real=1.0, 
    t::Matrix{<:Real}=time_matrix(duration=duration, delta_t=delta_t)
)
    λ = 1 / tau_s
    κ = mean_s * λ
    μ = 1 / tau_x
    ρ = μ * mean_x / mean_s
    GaussianSystem(κ, λ, ρ, μ, t=t)
end

function time_matrix(n::Integer, delta_t::Real)
    t_range = range(0.0, length=n, step=delta_t)
    transpose(t_range) .- t_range
end

function time_matrix(; duration::Real, delta_t::Real)
    n = round(Int, duration / delta_t, RoundUp)
    time_matrix(n, duration / n)
end


function marginal(system::GaussianSystem, t)
    MvNormal(LinearAlgebra.Symmetric(corr_xx(system).(t)))
end

function joint(system::GaussianSystem, t)
    MvNormal(LinearAlgebra.Symmetric(corr_z(system, t)))
end

function likelihood(system::GaussianSystem, t; signal::AbstractVector)
    c_ss = corr_ss(system).(t)
    c_sx = corr_sx(system).(t)
    c_xs = corr_xs(system).(t)
    c_xx = corr_xx(system).(t)
        
    regression_coef = c_sx * inv(c_ss)    
    p_x_given_s_cov = LinearAlgebra.Symmetric(c_xx - regression_coef * c_xs)
    
    MvNormal(regression_coef * signal, p_x_given_s_cov)
end

function posterior(system::GaussianSystem, t; response::AbstractVector)
    c_ss = corr_ss(system).(t)
    c_sx = corr_sx(system).(t)
    c_xs = corr_xs(system).(t)
    c_xx = corr_xx(system).(t)
        
    regression_coef = c_xs * inv(c_xx)    
    p_s_given_x_cov = LinearAlgebra.Symmetric(c_ss - regression_coef * c_sx)
    
    MvNormal(regression_coef * response, p_s_given_x_cov)
end

function log_likelihood(system::GaussianSystem, t; signal::AbstractArray, response::AbstractArray)
    c_ss = corr_ss(system).(t)
    c_sx = corr_sx(system).(t)
    c_xs = corr_xs(system).(t)
    c_xx = corr_xx(system).(t)
        
    regression_coef = c_sx * inv(c_ss)    
    p_x_given_s_cov = LinearAlgebra.Symmetric(c_xx - regression_coef * c_xs)
    
    scaled_normal = MvNormal(p_x_given_s_cov)
    
    logpdf(scaled_normal, response .- regression_coef * signal)
end

function log_joint(system::GaussianSystem, t; signal::AbstractArray, response::AbstractArray)
    c_z = corr_z(system, t)
    distr = MvNormal(c_z)
    logpdf(distr, vcat(signal, response))
end

function log_prior(system::GaussianSystem, t; signal::AbstractArray)
    distr = prior(system)
    logpdf(distr, signal)
end

struct GaussianChain{System <: GaussianSystem} <: MarkovChain
    scale::Float64
    θ::Float64
    system::System
end

GaussianChain(system::GaussianSystem; θ=1.0, scale::Real=1.0) = GaussianChain(scale, θ, system)
chain(system::GaussianSystem; θ=1.0, scale::Real=1.0) = GaussianChain(scale, θ, system)

function generate_configuration(system::GaussianSystem)
    rand(system.joint)
end

function new_signal(old_conf::Vector{Float64}, system::GaussianSystem)
    n_dim = length(old_conf) ÷ 2
    new_conf = copy(old_conf)
    new_conf[1:n_dim] .= rand(system.prior)
    new_conf
end

function propose!(new_state::Vector{Float64}, old_state::Vector{Float64}, kernel::GaussianChain)
    n_dim = length(old_state) ÷ 2
    new_state .= old_state
    normal_distr = Normal(0.0, kernel.scale)
    for i in 1:n_dim
        new_state[i] += rand(normal_distr)
    end
    new_state
end

energy(state::Vector{Float64}, chain::GaussianChain) = energy(state, chain.system, chain.θ)

function energy(state::Array{Float64,1}, system::GaussianSystem, θ=1.0)    
    result = 0.0
    n_dim = length(state) ÷ 2
    if θ > 0.0
        result += θ * logpdf(system.joint, state)
    end
    if (1.0 - θ) > 0.0
        result += (1.0 - θ) * logpdf(system.prior, @view state[1:n_dim])
    end
    -result
end

function energy_difference(state::Array{Float64,1}, system::GaussianSystem)
    energy(state, system, 1.0) - energy(state, system, 0.0)
end