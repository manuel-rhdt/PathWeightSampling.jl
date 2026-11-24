# The particle filter. The filter itself is the generic `SMC.sample`; this file defines the
# particle and the one-interval propagation.

# Advance one particle over SDE interval k: simulate the methylation jumps (exact, rates
# constant within the interval) and return the log-likelihood increment.
function step_interval!(m::AbstractVector{Int}, a::Vector{Float64}, logL, k, M::Model, rng)
    ℓ = logitL(M, logL)
    nc = length(m)
    A = 0.0
    Q = 0.0
    for i in 1:nc
        a[i] = act(M, m[i], ℓ)
        A += a[i]
        Q += exit_rate(M, m[i], a[i])
    end
    t0, t1 = (k - 1) * M.Δ, k * M.Δ
    t = t0 + randexp(rng) / Q
    t >= t1 && return output_loglik(M, A, M.nk[k], M.IYk[k])  # no jump (most intervals)
    lw = 0.0
    cur, Ncur, Fcur = t0, Ncum(M, t0), Fcum(M, t0)
    while true
        τ = min(t, t1)
        Nτ, Fτ = Ncum(M, τ), Fcum(M, τ)
        lw += output_loglik(M, A, Nτ - Ncur, Fτ - Fcur)
        τ == t1 && return lw
        # choose the jumping cluster and direction
        u = rand(rng) * Q
        c = 0
        for i in 1:nc
            q = exit_rate(M, m[i], a[i])
            if u <= q || i == nc
                c = i
                break
            end
            u -= q
        end
        mold = m[c]
        mnew = u <= up_rate(M, mold, a[c]) ? mold + 1 : mold - 1
        A -= a[c]; Q -= exit_rate(M, mold, a[c])
        m[c] = mnew
        a[c] = act(M, mnew, ℓ)
        A += a[c]; Q += exit_rate(M, mnew, a[c])
        cur, Ncur, Fcur = τ, Nτ, Fτ
        t = τ + randexp(rng) / Q
    end
end



"""
Setup data for the filters: the model, the fixed input signal (conditional case) or
`nothing` (marginal case: every particle carries its own signal), an optional reference
path (conditional SMC, see `bounds.jl`) and a scratch vector for the cluster activities.
`scratch` is shared by all particles, so one configuration must not be used from several
threads at once.
"""
struct FilterConfig{S,R}
    M::Model
    signal::S
    ref::R
    scratch::Vector{Float64}
    steps_per_level::Int
end

function FilterConfig(M::Model, signal, ref)
    spl = round(Int, M.dt / M.Δ)
    @assert spl * M.Δ ≈ M.dt
    FilterConfig(M, signal, ref, zeros(length(M.init)), spl)
end

"""
    Particle

A particle of the aggregated filter: the methylation level `m` of every cluster and, for
the marginal filter, the input signal state `(v, ll)` (velocity and log ligand). `lw` is the
log-weight increment of the last propagation. The fields `follow` and `ptr` are only used by
the conditional SMC, where a particle follows a reference path.
"""
mutable struct Particle <: SMC.AbstractParticle
    m::Vector{Int}
    v::Float64
    ll::Float64
    lw::Float64
    follow::Bool
    ptr::Int  # next reference jump (only used if `follow`)
end

SMC.weight(p::Particle) = p.lw

function SMC.spawn(::Type{Particle}, setup::SMC.Setup{<:FilterConfig})
    M = setup.configuration.M
    Particle(copy(M.init), 0.0, M.logc0, 0.0, false, 1)
end

# `follow` and `ptr` belong to the slot, not to the lineage, and are deliberately not copied.
function SMC.clone_from!(child::Particle, parent::Particle, ::SMC.Setup{<:FilterConfig})
    child.m .= parent.m
    child.v = parent.v
    child.ll = parent.ll
    child
end

"""Log-weight of a particle that follows the conditional-SMC reference over the SDE intervals `ks`."""
function follow_reference! end

function SMC.propagate!(p::Particle, tspan::Tuple{T,T}, setup::SMC.Setup{<:FilterConfig}) where {T<:Real}
    c = setup.configuration
    M, rng = c.M, setup.rng
    level = round(Int, tspan[1] / M.dt)
    ks = level*c.steps_per_level+1:(level+1)*c.steps_per_level
    if p.follow
        p.lw = follow_reference!(p, ks, c)
        return p
    end
    lw = 0.0
    for k in ks
        if c.signal === nothing
            lw += step_interval!(p.m, c.scratch, p.ll, k, M, rng)
            p.ll += log1p(M.gΔ * p.v)
            p.v = M.ρ * p.v + M.c * randn(rng)
        else
            lw += step_interval!(p.m, c.scratch, c.signal[k], k, M, rng)
        end
    end
    p.lw = lw
    p
end

"""
    run_smc(M, N, rng; signal=nothing) -> SMCResult

Particle filter with `N` particles on the aggregated likelihood. This is `SMC.sample` with
[`Particle`](@ref): weights are accumulated per output level (`M.dt`) and the population is
resampled (systematic) when the effective sample size falls below N/2. The result holds the
running estimate of ln P(x_{0:t}) and the effective sample size at t = 0, dt, …, T, including
the latent-independent part of the likelihood.

  * `signal === nothing`: marginal ln P(x); the particles carry (v, L, methylation).
  * `signal::Vector{Float64}` (from [`signal_logL`](@ref)): conditional ln P(x | s); the
    input is fixed and the particles carry only the methylation state.
"""
function run_smc(M::Model, N::Integer, rng; signal=nothing)
    setup = SMC.Setup(FilterConfig(M, signal, nothing), M, rng)
    particles = [SMC.spawn(Particle, setup) for _ in 1:N]
    log_marginal, ess = SMC.sample(setup, particles; rng=rng)
    SMC.SMCResult(collect(PWS.discrete_times(M)), log_marginal .+ M.Egrid, ess)
end
