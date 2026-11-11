"""
    ChemotaxisSMC

Fast particle filter (SMC) for the marginal and conditional probability of the output of
`PWS.chemotaxis_system`, with the output likelihood aggregated analytically.

The generic [`SMCEstimate`](@ref) propagates every reaction of every particle with the SSA,
including ~10^4 phosphorylation / dephosphorylation events per second. Here only the
methylation state of each receptor cluster (and, for the marginal, the input signal
`(v, log L)`) is simulated. The output reactions are *not* simulated: given the latent
state, the phosphorylation events are an inhomogeneous Poisson process with intensity
`k_A Y A(t)` (`A` = total kinase activity, piecewise constant between SDE steps and
methylation jumps), while the dephosphorylation events do not depend on the latent state.
Hence the log-weight of a particle over a time piece only needs the number of
phosphorylation events `n` in it and `∫ Y dt`:

    ln w = n ln A − k_A A ∫Y dt        (+ a latent-independent constant `Econst`)

This is the same weight as the SSA-based particle filter, but it is computed with a single
pass over the data, is independent of the output event rate, and uses exact (not
thinned) sampling of the methylation jumps between SDE steps. The signal-free path of the
algorithm (conditional probability, `signal` given) is also exact in the methylation
dynamics, so large particle numbers (10^5–10^6) are practical.

Beyond the point estimate, [`information_bounds`](@ref) gives stochastic lower *and* upper
bounds on the information density, using a conditional SMC that keeps the generating
(true) latent path as reference particle (bidirectional Monte Carlo). If the bounds
agree, the estimate has converged.
"""
module ChemotaxisSMC

export AggregatedSMCEstimate, information_bounds

import ..PathWeightSampling as PWS
import ..PathWeightSampling: AbstractSimulationAlgorithm, SimulationResult, discrete_times,
    log_marginal, to_dataframe, name
using ..JumpSystem: HybridJumpSystem, TraceAndTrajectory
using Random
using DataFrames
import LogExpFunctions: logsumexp

# ----------------------------------------------------------------------------------------
# Model and data

struct Model
    # time grid
    T::Float64
    Δ::Float64
    K::Int
    # SDE
    ρ::Float64       # 1 - λΔ
    c::Float64       # σ√Δ
    gΔ::Float64
    logc0::Float64
    # receptors
    KDa::Float64
    KDi::Float64
    N::Float64
    kR::Float64
    kB::Float64
    kA::Float64
    δf::Float64
    m0::Float64
    nm::Int                  # number of methylation states (0:nm-1)
    init::Vector{Int}        # initial state per cluster
    # data
    phos_t::Vector{Float64}  # phosphorylation event times in (0, T)
    brk_t::Vector{Float64}   # output event times (Y changes), with 0 first
    brk_F::Vector{Float64}   # ∫₀^{brk_t} Y dt
    brk_Y::Vector{Float64}   # Y just after brk_t
    nk::Vector{Int}          # phosphorylation events per SDE interval
    IYk::Vector{Float64}     # ∫Y dt per SDE interval
    Econst::Float64          # latent-independent part of ln P(x|·)
    dt::Float64              # output grid spacing (system.dt)
    Egrid::Vector{Float64}   # latent-independent part of ln P(x_{0:t}|·) at t = 0, dt, 2dt, …, T
end


Ncum(M::Model, t) = searchsortedlast(M.phos_t, t)
function Fcum(M::Model, t)
    j = searchsortedlast(M.brk_t, t)
    M.brk_F[j] + M.brk_Y[j] * (t - M.brk_t[j])
end

"""
    build_model(system, conf, T)

Extract parameters from `PWS.chemotaxis_system(...)` and the output data of configuration
`conf` (from `PWS.generate_configuration`) on [0, T].
"""
function build_model(system, conf, T)
    j = system.reactions
    λ, σ, g = system.sde_prob.p
    c0 = system.sde_prob.u0[2]
    @assert system.sde_prob.u0[1] == 0 "velocity must start at 0"
    Δ = system.sde_dt
    K = round(Int, T / Δ)
    nm = length(j.receptors)
    init = Int[]
    for (m, idx) in enumerate(j.receptors)
        append!(init, fill(m - 1, Int(system.u0[idx])))
    end

    trace = PWS.SSA.filter_trace(PWS.SSA.ReactionTrace(conf.trace), system.output_reactions)
    Y = Float64(system.u0[j.Y])
    Yp = Float64(system.u0[j.Yp])
    phos_t = Float64[]
    brk_t = [0.0]
    brk_F = [0.0]
    brk_Y = [Y]
    brk_E = [0.0]   # Σ log(rate) of output events up to brk_t
    brk_G = [0.0]   # ∫₀^{brk_t} Yp dt
    brk_Yp = [Yp]
    F = 0.0
    G = 0.0  # ∫ Yp dt
    tprev = 0.0
    Econst = 0.0
    for (τ, rx) in zip(trace.t, trace.rx)
        τ < T || break
        F += Y * (τ - tprev)
        G += Yp * (τ - tprev)
        tprev = τ
        type, _ = PWS.reaction_type(j, rx)
        if type == 2      # phosphorylation, rate k_A Y A
            Econst += log(j.k_A * Y)
            Y -= 1
            Yp += 1
            push!(phos_t, τ)
        elseif type == 3  # dephosphorylation, rate k_Z Yp
            Econst += log(j.k_Z * Yp)
            Y += 1
            Yp -= 1
        else
            error("unexpected output reaction $rx")
        end
        push!(brk_t, τ)
        push!(brk_F, F)
        push!(brk_Y, Y)
        push!(brk_E, Econst)
        push!(brk_G, G)
        push!(brk_Yp, Yp)
    end
    G += Yp * (T - tprev)
    Econst -= j.k_Z * G

    # latent-independent part on the output grid
    dt = system.dt
    nlev = round(Int, T / dt)
    @assert nlev * dt ≈ T "T must be a multiple of the output grid spacing $dt"
    Egrid = map(0:nlev) do i
        t = i * dt
        jj = searchsortedlast(brk_t, t)
        brk_E[jj] - j.k_Z * (brk_G[jj] + brk_Yp[jj] * (t - brk_t[jj]))
    end
    @assert Egrid[end] ≈ Econst

    M = Model(T, Δ, K, 1 - λ * Δ, σ * sqrt(Δ), g * Δ, log(c0),
        j.KD_a, j.KD_i, j.N, j.k_R, j.k_B, j.k_A, j.δf, j.m_0, nm, init,
        phos_t, brk_t, brk_F, brk_Y, Int[], Float64[], Econst, dt, Egrid)
    for k in 1:K
        t0, t1 = (k - 1) * Δ, k * Δ
        push!(M.nk, Ncum(M, t1) - Ncum(M, t0))
        push!(M.IYk, Fcum(M, t1) - Fcum(M, t0))
    end
    M
end

"""
    signal_logL(M, conf) -> Vector{Float64} (length M.K)

The input trajectory of `conf` as log L on each SDE interval, as used by the SMC:
entry k is the ligand on the SDE interval (t_{k-1}, t_k]. `conf.trace.u[i]` is (v, L) at time i·Δ, so
log L₁ = log c₀ and log L_{k+1} = log(u[k][2]).
"""
function signal_logL(M::Model, conf)
    tr = conf.trace
    @assert length(tr.u) >= M.K - 1
    @assert tr.dtimes[1] ≈ M.Δ && tr.dtimes[M.K-1] ≈ (M.K - 1) * M.Δ
    sig = Vector{Float64}(undef, M.K)
    sig[1] = M.logc0
    for k in 1:M.K-1
        sig[k+1] = log(tr.u[k][2])
    end
    sig
end


logitL(M::Model, logL) = M.N * (log1p(exp(logL) / M.KDa) - log1p(exp(logL) / M.KDi))
@inline act(M::Model, m, ℓ) = 1 / (1 + exp(-(ℓ - M.δf * (m - M.m0))))
@inline up_rate(M::Model, m, a) = m < M.nm - 1 ? (1 - a) * M.kR : 0.0
@inline down_rate(M::Model, m, a) = m > 0 ? a * M.kB : 0.0
@inline exit_rate(M::Model, m, a) = up_rate(M, m, a) + down_rate(M, m, a)

# ----------------------------------------------------------------------------------------
# SMC with the aggregated likelihood

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
    t >= t1 && return M.nk[k] * log(A) - M.kA * A * M.IYk[k]  # no jump (most intervals)
    lw = 0.0
    cur, Ncur, Fcur = t0, Ncum(M, t0), Fcum(M, t0)
    while true
        τ = min(t, t1)
        Nτ, Fτ = Ncum(M, τ), Fcum(M, τ)
        lw += (Nτ - Ncur) * log(A) - M.kA * A * (Fτ - Fcur)
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
    smc_run(M, N, rng; signal=nothing, out=nothing) -> ln P̂(x_{0:T})

Bootstrap particle filter with N particles; weights are accumulated per output level
(one level = `M.dt`, i.e. dt/Δ SDE steps) and the population is resampled (systematic) when
the ESS falls below N/2, exactly as in PWS' `SMC.sample`. Returns the full ln P̂ incl.
`M.Econst`.

  * `signal === nothing`: marginal ln P(x); the particles carry (v, L, methylation).
  * `signal::Vector{Float64}` (from `signal_logL`): conditional ln P(x | s); the input is
    fixed and the particles carry only the methylation state.
  * `out::Vector{Float64}` of length `length(M.Egrid)`: filled with the running estimate
    ln P̂(x_{0:t}) at t = 0, dt, …, T (out[1] = 0).
"""
function smc_run(M::Model, N::Int, rng; signal=nothing, out=nothing)
    spl = round(Int, M.dt / M.Δ)
    @assert spl * M.Δ ≈ M.dt
    out === nothing || @assert length(out) == length(M.Egrid)
    _smc_run(M, N, rng, spl, signal, out)
end

function _smc_run(M::Model, N::Int, rng, steps_per_level::Int,
        signal::Union{Nothing,Vector{Float64}}, out::Union{Nothing,Vector{Float64}})
    nc = length(M.init)
    V = zeros(N)
    LL = fill(M.logc0, N)
    Ms = repeat(M.init, 1, N)
    V2, LL2, Ms2 = similar(V), similar(LL), similar(Ms)
    logw = zeros(N)
    w = zeros(N)  # preallocated: the level loop must not allocate (GC pauses stall all threads)
    a = zeros(nc)
    nlev = M.K ÷ steps_per_level
    @assert nlev * steps_per_level == M.K
    out === nothing || (out[1] = 0.0)
    acc = 0.0
    for lev in 1:nlev
        for p in 1:N
            v, ll = V[p], LL[p]
            m = @view Ms[:, p]
            lw = 0.0
            for k in (lev-1)*steps_per_level+1:lev*steps_per_level
                if signal === nothing
                    lw += step_interval!(m, a, ll, k, M, rng)
                    ll += log1p(M.gΔ * v)
                    v = M.ρ * v + M.c * randn(rng)
                else
                    lw += step_interval!(m, a, signal[k], k, M, rng)
                end
            end
            V[p], LL[p] = v, ll
            logw[p] += lw
        end
        mx = maximum(logw)
        @. w = exp(logw - mx)
        out === nothing || (out[lev+1] = acc + mx + log(sum(w) / N) + M.Egrid[lev+1])
        lev == nlev && break
        ess = sum(w)^2 / sum(abs2, w)
        if ess < N / 2
            acc += mx + log(sum(w) / N)
            # systematic resampling
            W = sum(w)
            u = rand(rng) * W / N
            j = 1
            cum = w[1]
            for i in 1:N
                while cum < u && j < N  # guard: float roundoff in u
                    j += 1
                    cum += w[j]
                end
                V2[i], LL2[i] = V[j], LL[j]
                @views Ms2[:, i] .= Ms[:, j]
                u += W / N
            end
            V, V2 = V2, V
            LL, LL2 = LL2, LL
            Ms, Ms2 = Ms2, Ms
            logw .= 0
        end
    end
    acc + logsumexp(logw) - log(N) + M.Econst
end

# ----------------------------------------------------------------------------------------
# Conditional SMC with a fixed reference path (upper bound on ln P, bidirectional MC).
# If the reference is an exact posterior sample (the latent path that generated the data),
# E[1/Ẑ] = 1/Z, hence E[ln Ẑ] ≥ ln Z: ln Ẑ_csmc is a stochastic upper bound.

"""Methylation jumps of the generating path, with cluster labels: time, cluster, new state."""
struct RefPath
    jt::Vector{Float64}
    jc::Vector{Int}
    jm::Vector{Int}
end

"""
    true_path(M, system, conf, rng) -> RefPath

The methylation jumps of the simulation that produced `conf`. PWS tracks only the number of
clusters per methylation state; each jump is assigned to a cluster chosen uniformly among
those in the source state, which gives an exact sample of the labelled path.
"""
function true_path(M::Model, system, conf, rng)
    j = system.reactions
    rt = PWS.SSA.ReactionTrace(conf.trace)
    m = copy(M.init)
    jt = Float64[]; jc = Int[]; jm = Int[]
    for (τ, rx) in zip(rt.t, rt.rx)
        τ < M.T || break
        ty, mold = PWS.reaction_type(j, rx)
        ty in (0, 1) || continue
        @assert (ty == 0) == (rx <= M.nm) "unexpected methylation reaction numbering"
        cands = findall(==(mold), m)
        @assert !isempty(cands) "no cluster in state $mold at t=$τ"
        c = cands[rand(rng, 1:length(cands))]
        m[c] = ty == 0 ? mold + 1 : mold - 1
        push!(jt, τ); push!(jc, c); push!(jm, m[c])
    end
    RefPath(jt, jc, jm)
end

# likelihood increment of SDE interval k along the reference jumps; returns (lw, next jump index)
function ref_interval!(m::AbstractVector{Int}, a::Vector{Float64}, logL, k, M::Model, R::RefPath, ptr::Int)
    ℓ = logitL(M, logL)
    A = 0.0
    for i in eachindex(m)
        a[i] = act(M, m[i], ℓ)
        A += a[i]
    end
    t0, t1 = (k - 1) * M.Δ, k * M.Δ
    lw = 0.0
    Ncur, Fcur = Ncum(M, t0), Fcum(M, t0)
    while ptr <= length(R.jt) && R.jt[ptr] <= t1
        τ = R.jt[ptr]
        Nτ, Fτ = Ncum(M, τ), Fcum(M, τ)
        lw += (Nτ - Ncur) * log(A) - M.kA * A * (Fτ - Fcur)
        c = R.jc[ptr]
        A -= a[c]
        m[c] = R.jm[ptr]
        a[c] = act(M, m[c], ℓ)
        A += a[c]
        Ncur, Fcur = Nτ, Fτ
        ptr += 1
    end
    Nτ, Fτ = Ncum(M, t1), Fcum(M, t1)
    lw += (Nτ - Ncur) * log(A) - M.kA * A * (Fτ - Fcur)
    lw, ptr
end

"""
    csmc_run(M, N, rng, R, refsig; signal=nothing) -> (ln Ẑ, ref log-likelihood)

Conditional bootstrap SMC: particle 1 follows the reference (methylation jumps `R`, signal
`refsig`); the other N-1 are resampled multinomially from all N at every output level.
With `signal` (the true signal, = refsig) the signal is fixed for all particles (conditional
density); without it the other particles sample the signal from the prior (marginal).
Returns the estimate of ln P(x_{0:T}) (an upper bound in expectation) and, as a check, the
reference's own log-likelihood ln P(x_{0:T} | reference path) (both include Econst).
"""
function csmc_run(M::Model, N::Int, rng, R::RefPath, refsig::Vector{Float64}; signal=nothing)
    spl = round(Int, M.dt / M.Δ)
    nc = length(M.init)
    V = zeros(N); LL = fill(M.logc0, N); Ms = repeat(M.init, 1, N)
    V2, LL2, Ms2 = similar(V), similar(LL), similar(Ms)
    logw = zeros(N); w = zeros(N); a = zeros(nc)
    nlev = M.K ÷ spl
    @assert nlev * spl == M.K
    acc = 0.0
    refll = 0.0
    ptr = 1
    for lev in 1:nlev
        ks = (lev-1)*spl+1:lev*spl
        # reference particle
        lw = 0.0
        m1 = @view Ms[:, 1]
        for k in ks
            l, ptr = ref_interval!(m1, a, refsig[k], k, M, R, ptr)
            lw += l
        end
        logw[1] = lw
        refll += lw
        for p in 2:N
            v, ll = V[p], LL[p]
            m = @view Ms[:, p]
            lw = 0.0
            for k in ks
                if signal === nothing
                    lw += step_interval!(m, a, ll, k, M, rng)
                    ll += log1p(M.gΔ * v)
                    v = M.ρ * v + M.c * randn(rng)
                else
                    lw += step_interval!(m, a, signal[k], k, M, rng)
                end
            end
            V[p], LL[p] = v, ll
            logw[p] = lw
        end
        mx = maximum(logw)
        @. w = exp(logw - mx)
        W = sum(w)
        acc += mx + log(W / N)
        lev == nlev && break
        # multinomial resampling of slots 2..N from all N (sorted uniforms by spacings)
        E = randexp(rng, N)
        tot = sum(E)
        cum = 0.0
        u = 0.0
        j = 1
        cw = w[1]
        for i in 2:N
            u += E[i-1]
            target = u / tot * W
            while cw < target && j < N
                j += 1
                cw += w[j]
            end
            V2[i], LL2[i] = V[j], LL[j]
            @views Ms2[:, i] .= Ms[:, j]
        end
        V2[1], LL2[1] = V[1], LL[1]
        @views Ms2[:, 1] .= Ms[:, 1]
        V, V2 = V2, V; LL, LL2 = LL2, LL; Ms, Ms2 = Ms2, Ms
    end
    @assert ptr == length(R.jt) + 1 "reference jumps not all consumed"
    acc + M.Econst, refll + M.Econst
end

# ----------------------------------------------------------------------------------------
# Integration with the PWS algorithm interface

"""
    AggregatedSMCEstimate(num_particles::Int)

Particle-filter estimate of the marginal and conditional trajectory probability for
`PWS.chemotaxis_system`, using the aggregated output likelihood (see [`ChemotaxisSMC`](@ref)).
It estimates the same quantities as [`SMCEstimate`](@ref) but is much faster for this
system and therefore allows far larger particle numbers.

```julia
system = PWS.chemotaxis_system(n_clusters=25, duration=20.0)
mi = PWS.mutual_information(system, PWS.AggregatedSMCEstimate(2^16); num_samples=8)
```
"""
struct AggregatedSMCEstimate <: AbstractSimulationAlgorithm
    num_particles::Int
end

name(x::AggregatedSMCEstimate) = "AggSMC-$(x.num_particles)"

struct AggregatedSMCResult <: SimulationResult
    time::Vector{Float64}
    log_marginal_estimate::Vector{Float64}
end

log_marginal(result::AggregatedSMCResult) = result.log_marginal_estimate
to_dataframe(result::AggregatedSMCResult) =
    DataFrame(:time => result.time, :log_marginal => result.log_marginal_estimate)

const ChemotaxisSystem = HybridJumpSystem{<:Any,<:PWS.ChemotaxisJumps}

function _run(system, algorithm::AggregatedSMCEstimate, conf, rng; conditional)
    M = build_model(system, conf, system.tspan[2])
    out = zeros(length(M.Egrid))
    smc_run(M, algorithm.num_particles, rng; signal=conditional ? signal_logL(M, conf) : nothing, out=out)
    AggregatedSMCResult(Vector(discrete_times(system)), out)
end

function PWS.marginal_density(system::ChemotaxisSystem, algorithm::AggregatedSMCEstimate, conf::TraceAndTrajectory; full_result=false, rng=Random.default_rng(), kwargs...)
    result = _run(system, algorithm, conf, rng; conditional=false)
    full_result ? result : log_marginal(result)
end

function PWS.conditional_density(system::ChemotaxisSystem, algorithm::AggregatedSMCEstimate, conf::TraceAndTrajectory; full_result=false, rng=Random.default_rng(), kwargs...)
    result = _run(system, algorithm, conf, rng; conditional=true)
    full_result ? result : log_marginal(result)
end

"""
    information_bounds(system, conf, num_particles; rng) -> NamedTuple

Stochastic lower and upper bounds on the trajectory information density
`ln P(x|s) − ln P(x)` of the configuration `conf` (evaluated at the end of the trajectory),
for `system = PWS.chemotaxis_system(...)`. In expectation over the SMC randomness

    lower ≤ ln P(x|s) − ln P(x) ≤ upper,

where the lower bound uses the ordinary particle filter for ln P(x|s) and a conditional
particle filter (keeping the true latent path as reference) for ln P(x), and the upper
bound the reverse. If the two agree within noise the particle number is sufficient.
The result also contains the plain estimate `smc = ln P̂(x|s) − ln P̂(x)`.
"""
function information_bounds(system::ChemotaxisSystem, conf::TraceAndTrajectory, num_particles::Integer; rng=Random.default_rng())
    M = build_model(system, conf, system.tspan[2])
    sig = signal_logL(M, conf)
    R = true_path(M, system, conf, rng)
    N = num_particles
    cond_smc = smc_run(M, N, rng; signal=sig)
    cond_csmc = csmc_run(M, N, rng, R, sig; signal=sig)[1]
    marg_smc = smc_run(M, N, rng)
    marg_csmc = csmc_run(M, N, rng, R, sig)[1]
    (lower=cond_smc - marg_csmc, smc=cond_smc - marg_smc, upper=cond_csmc - marg_smc,
        cond_smc=cond_smc, cond_csmc=cond_csmc, marg_smc=marg_smc, marg_csmc=marg_csmc)
end

end # module
