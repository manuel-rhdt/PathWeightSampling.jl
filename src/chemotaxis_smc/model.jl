# Model: the chemotaxis system parameters together with the output data of one configuration.

"""
    Model

Everything the filter needs: the jump set of the system (receptor and CheY parameters), the
time grids, the input-signal recursion and the output data of one trajectory in aggregated
form. Build it with [`build_model`](@ref).

The input signal on the SDE grid (step `Δ = system.sde_dt`) follows the Euler–Maruyama
recursion of the system,

    v₁ = 0,   v_{k+1} = ρ v_k + c ξ_k,      log L₁ = log c₀,   log L_{k+1} = log L_k + log(1 + gΔ v_k),

with ξ_k standard normal. The ligand is constant on each SDE interval.
"""
struct Model
    jumps::PWS.ChemotaxisJumps
    # time grids
    T::Float64
    Δ::Float64               # SDE step
    K::Int                   # number of SDE intervals
    dt::Float64              # output grid spacing (system.dt)
    # input signal recursion
    ρ::Float64               # 1 - λΔ
    c::Float64               # σ√Δ
    gΔ::Float64
    logc0::Float64
    init::Vector{Int}        # initial methylation level of every cluster
    # output data: phosphorylation events and the integral of Y
    phos_t::Vector{Float64}  # phosphorylation event times in (0, T)
    brk_t::Vector{Float64}   # output event times (Y changes), with 0 first
    brk_F::Vector{Float64}   # ∫₀^{brk_t} Y dt
    brk_Y::Vector{Float64}   # Y just after brk_t
    nk::Vector{Int}          # phosphorylation events per SDE interval
    IYk::Vector{Float64}     # ∫Y dt per SDE interval
    Econst::Float64          # latent-independent part of ln P(x|·)
    Egrid::Vector{Float64}   # latent-independent part of ln P(x_{0:t}|·) at t = 0, dt, 2dt, …, T
end

nstates(M::Model) = length(M.jumps.receptors)  # number of methylation levels

Ncum(M::Model, t) = searchsortedlast(M.phos_t, t)  # phosphorylation events up to t
function Fcum(M::Model, t)                         # ∫₀ᵗ Y dt
    j = searchsortedlast(M.brk_t, t)
    M.brk_F[j] + M.brk_Y[j] * (t - M.brk_t[j])
end

PWS.discrete_times(M::Model) = range(0.0, M.T, step=M.dt)

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

    M = Model(j, T, Δ, K, dt, 1 - λ * Δ, σ * sqrt(Δ), g * Δ, log(c0), init,
        phos_t, brk_t, brk_F, brk_Y, Int[], Float64[], Econst, Egrid)
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

# Receptor-cluster activity and methylation rates. `ℓ` is the ligand-dependent part of the
# free energy difference, `logitL`; a cluster at methylation level `m` is active with
# probability `act(M, m, ℓ)`.
logitL(M::Model, logL) = M.jumps.N * (log1p(exp(logL) / M.jumps.KD_a) - log1p(exp(logL) / M.jumps.KD_i))
@inline act(M::Model, m, ℓ) = 1 / (1 + exp(-(ℓ - M.jumps.δf * (m - M.jumps.m_0))))
@inline up_rate(M::Model, m, a) = m < nstates(M) - 1 ? (1 - a) * M.jumps.k_R : 0.0
@inline down_rate(M::Model, m, a) = m > 0 ? a * M.jumps.k_B : 0.0
@inline exit_rate(M::Model, m, a) = up_rate(M, m, a) + down_rate(M, m, a)

# Log-likelihood of `n` phosphorylation events with `∫Y dt = IY` while the total kinase
# activity is `A`: the events are Poisson with intensity k_A Y A.
@inline output_loglik(M::Model, A, n, IY) = n * log(A) - M.jumps.k_A * A * IY
