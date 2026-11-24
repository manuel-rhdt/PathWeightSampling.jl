# Convergence diagnostics for the particle filter: a conditional SMC that keeps the true latent
# path as a reference particle gives a stochastic *upper* bound on ln P (bidirectional Monte
# Carlo), the ordinary filter a stochastic *lower* bound. If lower and upper bound of the
# information density agree, the particle number is sufficient.

"""Methylation jumps of the generating path (time, cluster, new state) and its signal."""
struct RefPath
    jt::Vector{Float64}
    jc::Vector{Int}
    jm::Vector{Int}
    v::Vector{Float64}      # velocity on the SDE grid
    logL::Vector{Float64}   # log ligand on the SDE intervals (`signal_logL`)
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
        @assert (ty == 0) == (rx <= nstates(M)) "unexpected methylation reaction numbering"
        cands = findall(==(mold), m)
        @assert !isempty(cands) "no cluster in state $mold at t=$τ"
        c = cands[rand(rng, 1:length(cands))]
        m[c] = ty == 0 ? mold + 1 : mold - 1
        push!(jt, τ); push!(jc, c); push!(jm, m[c])
    end
    sig = signal_logL(M, conf)
    v = [0.0; [conf.trace.u[k][1] for k in 1:M.K-1]]
    RefPath(jt, jc, jm, v, sig)
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
        lw += output_loglik(M, A, Nτ - Ncur, Fτ - Fcur)
        c = R.jc[ptr]
        A -= a[c]
        m[c] = R.jm[ptr]
        a[c] = act(M, m[c], ℓ)
        A += a[c]
        Ncur, Fcur = Nτ, Fτ
        ptr += 1
    end
    Nτ, Fτ = Ncum(M, t1), Fcum(M, t1)
    lw += output_loglik(M, A, Nτ - Ncur, Fτ - Fcur)
    lw, ptr
end

# The reference particle does not sample: it follows the reference jumps and carries the true
# signal state, which particles cloned from it inherit.
function follow_reference!(p::Particle, ks, c::FilterConfig{<:Any,<:RefPath})
    M, R = c.M, c.ref
    lw = 0.0
    for k in ks
        l, p.ptr = ref_interval!(p.m, c.scratch, R.logL[k], k, M, R, p.ptr)
        lw += l
    end
    if last(ks) < M.K
        p.v, p.ll = R.v[last(ks)+1], R.logL[last(ks)+1]
    end
    lw
end

"""
    run_csmc(M, N, rng, R; conditional=false) -> (result, reference_loglik)

Conditional particle filter: particle 1 follows the reference `R` (the generating
methylation jumps and signal); the other N-1 are resampled multinomially from all N at
every output level (`SMC.sample` with `resample_threshold=Inf` and
`SMC.multinomial_sample_keep_first`). With `conditional=true` the signal is fixed to the
reference signal (conditional density ln P(x|s)); otherwise the other particles sample the
signal from the prior (marginal ln P(x)).

Returns an `SMCResult` like [`run_smc`](@ref) whose estimate is an upper bound in
expectation (the reference is an exact posterior sample) and, as a check, the reference's own
log-likelihood ln P(x_{0:T} | reference path).
"""
function run_csmc(M::Model, N::Integer, rng, R::RefPath; conditional=false)
    setup = SMC.Setup(FilterConfig(M, conditional ? R.logL : nothing, R), M, rng)
    particles = [SMC.spawn(Particle, setup) for _ in 1:N]
    particles[1].follow = true
    refll = 0.0
    log_marginal, ess = SMC.sample(setup, particles; rng=rng, resample_threshold=Inf,
        resampler=SMC.multinomial_sample_keep_first, inspect=ps -> (refll += ps[1].lw))
    @assert particles[1].ptr == length(R.jt) + 1 "reference jumps not all consumed"
    SMC.SMCResult(collect(PWS.discrete_times(M)), log_marginal .+ M.Egrid, ess), refll + M.Econst
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
The result also contains the plain estimate `smc = ln P̂(x|s) − ln P̂(x)` and the four
log-density estimates it is built from.
"""
function information_bounds(system::ChemotaxisSystem, conf::TraceAndTrajectory, num_particles::Integer; rng=Random.default_rng())
    M = build_model(system, conf, system.tspan[2])
    R = true_path(M, system, conf, rng)
    N = num_particles
    final(result) = last(log_marginal(result))
    cond_smc = final(run_smc(M, N, rng; signal=R.logL))
    cond_csmc = final(run_csmc(M, N, rng, R; conditional=true)[1])
    marg_smc = final(run_smc(M, N, rng))
    marg_csmc = final(run_csmc(M, N, rng, R)[1])
    (lower=cond_smc - marg_csmc, smc=cond_smc - marg_smc, upper=cond_csmc - marg_smc,
        cond_smc=cond_smc, cond_csmc=cond_csmc, marg_smc=marg_smc, marg_csmc=marg_csmc)
end
