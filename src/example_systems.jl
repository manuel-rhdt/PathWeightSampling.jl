import Catalyst
import Catalyst: @reaction_network, @reaction
import ModelingToolkit
import ModelingToolkit: FnType

"""
    gene_expression_system(; mean_s=50, mean_x=mean_s, corr_time_s=1.0, corr_time_x=0.1, u0=SA[mean_s, mean_x], dtimes=0:0.1:2.0)

Creates a system for a very simple model of gene expression.
    
# Model Description

This model consists of four reactions:
```
                κ
reaction 1:  ∅ ---> S

                λ
reaction 2:  S ---> ∅

                ρ
reaction 3:  S ---> S + X

                μ
reaction 4:  X ---> ∅
```

The first two reactions specify the evolution of the input signal `S`,
and last two reactions specify the evolution of the output `X`. Thus,
both the input and output signal are modeled as a simple birth-death 
process, however the birth rate of `X` increases with higher copy numbers 
of  `S`.

# Examples

The values of the reaction rates can be specified directly as follows:
```jldoctest
using PathWeightSampling
system = PathWeightSampling.gene_expression_system(kappa = 10.0, lambda = 0.1, rho = 1.0, mu = 1.0)

# output

SimpleSystem with 4 reactions
Input variables: S(t)
Output variables: X(t)
Initial condition:
    S(t) = 50
    X(t) = 50
Parameters:
    κ = 10.0
    λ = 0.1
    ρ = 1.0
    μ = 1.0
```

Alternatively, the reaction rates can be specified indirectly through the following arguments:
- `mean_s`: The average copy number of S when the system has relaxed to steady state.
- `mean_x`: The average copy number of X when the system has relaxed to steady state.
- `corr_time_s`: The input signal correlation time. The shorter this time is, the faster the fluctuations in the input signal.
- `corr_time_x`: The output signal correlation time. This sets the timescale of output fluctuations.

```jldoctest
using PathWeightSampling
system = PathWeightSampling.gene_expression_system(mean_s=25, mean_x=50, corr_time_s=1.0, corr_time_x=0.75)

# output

SimpleSystem with 4 reactions
Input variables: S(t)
Output variables: X(t)
Initial condition:
    S(t) = 25
    X(t) = 50
Parameters:
    κ = 25.0
    λ = 1.0
    ρ = 2.666666666666666
    μ = 1.3333333333333333
```
"""
function gene_expression_system(;
    mean_s=50,
    mean_x=mean_s,
    corr_time_s=1.0,
    corr_time_x=0.1,
    kappa=mean_s / corr_time_s,
    lambda=1 / corr_time_s,
    mu=1 / corr_time_x,
    rho=mu * mean_x / mean_s,
    u0=SA[mean_s, mean_x],
    dtimes=0:0.1:2.0
)
    sn = @reaction_network begin
        κ, ∅ --> S
        λ, S --> ∅
    end κ λ

    xn = @reaction_network begin
        ρ, S --> X + S
        μ, X --> ∅
    end ρ μ

    ps = [kappa, lambda]
    px = [rho, mu]

    SimpleSystem(sn, xn, u0, ps, px, dtimes)
end

function chemotaxis_system(;
    mean_L=20,
    num_receptors=100,
    Y_tot=500,
    L_timescale=1.0,
    LR_timescale=0.01,
    LR_ratio=0.5,
    Y_timescale=0.1,
    Y_ratio=1 / 2,
    q=0,
    dtimes=0:0.1:2.0
)
    mean_LR = num_receptors * LR_ratio
    mean_R = num_receptors - mean_LR

    mean_Yp = Y_tot * Y_ratio
    mean_Y = Y_tot - mean_Yp

    eq_L = mean_L * exp(-q)

    sn = @reaction_network begin
        κ, ∅ --> L
        λ, L --> ∅
    end κ λ

    rn = @reaction_network begin
        ρ, L + R --> L + LR
        μ, LR --> R
    end ρ μ

    xn = @reaction_network begin
        δ, LR + Y --> Yp + LR
        χ, Yp --> Y
    end δ χ

    u0 = round.([mean_L, mean_R, mean_LR, mean_Y, mean_Yp])
    ps = [mean_L, 1 / L_timescale]

    ρ = inv(eq_L * LR_timescale * (1 + mean_R / mean_LR))
    μ = eq_L * ρ * mean_R / mean_LR

    pr = [ρ, μ]

    χ = inv(Y_timescale * (1 + Y_ratio))
    δ = χ * Y_ratio / mean_LR

    px = [δ, χ]

    ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes)
end

function chemotaxis_parameters(;
    E₀=3.0,
    Kₐ=500,
    Kᵢ=25,
    δg=log(Kₐ / Kᵢ),
    δf=-1.5,
    k⁺=0.05,
    k⁺ₐ=k⁺,
    k⁺ᵢ=k⁺,
    k⁻ₐ=Kₐ * k⁺ₐ,
    k⁻ᵢ=Kᵢ * k⁺ᵢ,
    a_star=0.5,
    γ=1 / 10,
    k_B=(1 - a_star) * γ / abs(δf),
    k_R=a_star * γ / abs(δf)
)
    [
        E₀,
        δg,
        δf,
        k⁺ₐ,
        k⁺ᵢ,
        k⁻ₐ,
        k⁻ᵢ,
        k_B,
        k_R
    ]
end

"""
    cooperative_chemotaxis_system(;
        lmax = 3,
        mmax = 9,
        n_clusters = 100,
        n_chey = 10_000,

        mean_l = 50,
        tau_l = 1.0,

        phosphorylate = 2000.0 / (n_chey * n_clusters),
        dephosphorylate = 2000.0 / (n_chey),
        dtimes = 0:0.1:20.0,
        varargs...
    )

Create a system for a complex chemotaxis model.

# Model Description

This model describes the bacterial chemotaxis signaling network.



```
Rml -> Rm(l+1), with rate kon L(t) (lmax - l): ligand binding to active state
Rm(l+1) -> Rml, with rate koff_A l: ligand unbinding
Rml -> R(m+1)l with rate km: methylation rate
R(m+1)l -> Rml with rate kdm: demethylation rate
```

# Examples

Create a chemotaxis system with default parameters.
```jldoctest
using PathWeightSampling
PathWeightSampling.cooperative_chemotaxis_system()

# output

ComplexSystem with 175 reactions
Input variables: L(t)
Latent variables: R_0_0(t), R_0_1(t), R_0_2(t), R_0_3(t), R_0_4(t), R_0_5(t), R_0_6(t), R_0_7(t), R_0_8(t), R_0_9(t), R_1_0(t), R_1_1(t), R_1_2(t), R_1_3(t), R_1_4(t), R_1_5(t), R_1_6(t), R_1_7(t), R_1_8(t), R_1_9(t), R_2_0(t), R_2_1(t), R_2_2(t), R_2_3(t), R_2_4(t), R_2_5(t), R_2_6(t), R_2_7(t), R_2_8(t), R_2_9(t), R_3_0(t), R_3_1(t), R_3_2(t), R_3_3(t), R_3_4(t), R_3_5(t), R_3_6(t), R_3_7(t), R_3_8(t), R_3_9(t)
Output variables: Yp(t), Y(t)
Initial condition:
    L(t) = 50
    R_0_0(t) = 100
    R_0_1(t) = 0
    R_0_2(t) = 0
    R_0_3(t) = 0
    R_0_4(t) = 0
    R_0_5(t) = 0
    R_0_6(t) = 0
    R_0_7(t) = 0
    R_0_8(t) = 0
    R_0_9(t) = 0
    R_1_0(t) = 0
    R_1_1(t) = 0
    R_1_2(t) = 0
    R_1_3(t) = 0
    R_1_4(t) = 0
    R_1_5(t) = 0
    R_1_6(t) = 0
    R_1_7(t) = 0
    R_1_8(t) = 0
    R_1_9(t) = 0
    R_2_0(t) = 0
    R_2_1(t) = 0
    R_2_2(t) = 0
    R_2_3(t) = 0
    R_2_4(t) = 0
    R_2_5(t) = 0
    R_2_6(t) = 0
    R_2_7(t) = 0
    R_2_8(t) = 0
    R_2_9(t) = 0
    R_3_0(t) = 0
    R_3_1(t) = 0
    R_3_2(t) = 0
    R_3_3(t) = 0
    R_3_4(t) = 0
    R_3_5(t) = 0
    R_3_6(t) = 0
    R_3_7(t) = 0
    R_3_8(t) = 0
    R_3_9(t) = 0
    Yp(t) = 0
    Y(t) = 10000
Parameters:
    κ = 50.0
    λ = 1.0
    E0 = 3.0
    δg = 2.995732273553991
    δf = -1.5
    lba = 0.05
    lbi = 0.05
    lda = 25.0
    ldi = 1.25
    mda = 0.03333333333333333
    mbi = 0.03333333333333333
    μ = 8.57142857142857
    ρ = 0.028571428571428564
```

"""
function cooperative_chemotaxis_system(;
    lmax=3,
    mmax=9,
    n_clusters=100,
    n_chey=10_000,
    mean_l=50,
    tau_l=1.0,
    tau_y=0.1,
    phi_y=1.0 / 6.0,
    dephosphorylate=inv(tau_y * (1 + phi_y)),
    phosphorylate=dephosphorylate * phi_y / (n_clusters / 2),
    dtimes=0:0.1:20.0, aggregator=JumpProcesses.SortingDirect(),
    dist_aggregator=DepGraphDirect(),
    varargs...
)
    sn = @reaction_network begin
        κ, ∅ --> L
        λ, L --> ∅
    end κ λ

    rn = Catalyst.make_empty_network()

    Catalyst.@parameters t E0 δg δf lba lbi lda ldi mda mbi ρ
    Catalyst.@variables L(t) Y(t) Yp(t)

    Catalyst.addspecies!(rn, L)

    Catalyst.addparam!(rn, E0)
    Catalyst.addparam!(rn, δg)
    Catalyst.addparam!(rn, δf)
    Catalyst.addparam!(rn, lba)
    Catalyst.addparam!(rn, lbi)
    Catalyst.addparam!(rn, lda)
    Catalyst.addparam!(rn, ldi)
    Catalyst.addparam!(rn, mda)
    Catalyst.addparam!(rn, mbi)

    xn = @reaction_network begin
        μ, Yp --> Y
    end μ

    Catalyst.addparam!(xn, ρ)

    spmap = Dict()
    for l = 0:lmax, m = 0:mmax
        receptor_species = ModelingToolkit.Num(ModelingToolkit.variable(Symbol("R_", l, "_", m), T=FnType{Tuple{Any},Real}))(t)

        spmap[(l, m)] = receptor_species

        Catalyst.addspecies!(xn, receptor_species)
        Catalyst.addspecies!(rn, receptor_species)
    end

    p_active(l, m) = 1 / (1 + exp(E0 + l * δg + m * δf))

    for l = 0:lmax, m = 0:mmax

        if l > 0
            ligand_bind = Catalyst.Reaction((lba * p_active(l - 1, m) + lbi * (1 - p_active(l - 1, m))) * (lmax + 1 - l), [spmap[(l - 1, m)], L], [spmap[(l, m)], L])
            ligand_unbind = Catalyst.Reaction((lda * p_active(l, m) + ldi * (1 - p_active(l, m))) * l, [spmap[(l, m)]], [spmap[(l - 1, m)]])
            Catalyst.addreaction!(rn, ligand_bind)
            Catalyst.addreaction!(rn, ligand_unbind)
        end

        if m > 0
            demethylate_active = Catalyst.Reaction(mda * p_active(l, m), [spmap[(l, m)]], [spmap[(l, m - 1)]])
            methylate_inactive = Catalyst.Reaction(mbi * (1 - p_active(l, m - 1)), [spmap[(l, m - 1)]], [spmap[(l, m)]])

            Catalyst.addreaction!(rn, demethylate_active)
            Catalyst.addreaction!(rn, methylate_inactive)
        end


        # every receptor phosphorylates Y with rate ρ if the receptor is active
        # if the receptor is inactive, no phosphorylation can happen
        receptor = spmap[(l, m)]
        phosphorylation = Catalyst.Reaction(ρ * p_active(l, m), [Y, receptor], [Yp, receptor])
        Catalyst.addreaction!(xn, phosphorylation)
    end

    joint = ModelingToolkit.extend(xn, ModelingToolkit.extend(rn, sn))

    ps = [mean_l / tau_l, 1.0 / tau_l]
    pr = chemotaxis_parameters(; varargs...)
    px = [dephosphorylate, phosphorylate]

    spec2index = Catalyst.speciesmap(joint)
    u0 = zeros(Int, length(spmap) + 3) # (extra space +3 for input and both outputs)
    u0[1] = round(Int, mean_l)
    u0[spec2index[Yp]] = round(Int, n_chey * phi_y)
    u0[spec2index[Y]] = n_chey - u0[spec2index[Yp]]

    # here we approximate the steady states of the individual receptors
    E_0 = pr[1]
    δg = pr[2]
    δf = pr[3]
    k_B = pr[8]
    k_R = pr[9]
    K_a = pr[6] / pr[4]
    K_i = pr[7] / pr[5]

    l = 0:lmax
    m = 0:mmax
    a_m = activity_given_methylation.(m, c=mean_l, N=lmax, K_a=K_a, K_i=K_i, δfₘ=δf, m₀=-E_0 / δf)
    p_m = steady_state_methylation(a_m, k_B=k_B, k_R=k_R)
    p_l_m = ligand_given_methylation(l, a_m, c=mean_l, N=lmax, K_a=K_a, K_i=K_i) .* p_m'

    for state in systematic_sample(vec(p_l_m); N=n_clusters)
        l_i, m_i = Tuple(CartesianIndices((l, m))[state])
        u0[spec2index[spmap[(l_i, m_i)]]] += 1
    end

    ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes; aggregator=aggregator, dist_aggregator=dist_aggregator)
end

function steady_state_methylation(a_m; k_R=0.075, k_B=0.15)
    length(a_m) == 0 && return Float64[]
    length(a_m) == 1 && return [1.0]
    dl = @. (1 - a_m[1:end-1]) * k_R
    d = @. -(1 - a_m) * k_R - a_m * k_B
    du = @. a_m[2:end] * k_B

    Q_mat = Tridiagonal(dl, d, du)
    vcat(Q_mat, ones(1, length(a_m))) \ vcat(zeros(length(a_m)), 1.0)
end

function activity_given_methylation(m; N=6, K_a=2000, K_i=20, δfₘ=-2.0, m₀=0.5 * N, c=100)
    E_act = δfₘ * (m - m₀) # energy from methylation
    Z_a = exp(-E_act) * (1 + c / K_a)^N
    Z_i = (1 + c / K_i)^N
    Z_a / (Z_a + Z_i)
end

function ligand_given_methylation(ℓ, a_m; N=6, K_a=2000, K_i=20, c=100)
    Z_a = (1 + c / K_a)^N
    Z_i = (1 + c / K_i)^N
    [a * binomial(N, ℓ) * (c / K_a)^ℓ / Z_a + (1 - a) * binomial(N, ℓ) * (c / K_i)^ℓ / Z_i for ℓ ∈ ℓ, a ∈ a_m]
end

function sde_chemotaxis_system(;
    velocity_decay=0.862,
    velocity_noise=sqrt(2 * velocity_decay * 157.1),
    gradient_steepness=0.2e-3,
    harmonic_rate=nothing,
    kwargs...
)
    ModelingToolkit.@variables t V(t) L(t)
    ModelingToolkit.@parameters λ σ g ω₀

    D = ModelingToolkit.Differential(t)

    if harmonic_rate === nothing
        eqs = [
            D(V) ~ -λ * V,
            D(L) ~ g * L * V
        ]
        noiseeqs = [σ, 0]
        sparams = [λ, σ, g]
        ps = [velocity_decay, velocity_noise, gradient_steepness]
    else
        c₀ = 100.0 # μM
        eqs = [
            D(V) ~ -ω₀^2 * (L - c₀) - λ * V,
            D(L) ~ g * c₀ * V
        ]
        noiseeqs = [σ, 0]
        sparams = [λ, σ, g, ω₀]
        ps = [velocity_decay, velocity_noise, gradient_steepness, harmonic_rate]
    end

    ModelingToolkit.@named sn = SDESystem(eqs, noiseeqs, t, [V, L], sparams)

    system = PathWeightSampling.cooperative_chemotaxis_system(; kwargs...)

    Catalyst.addspecies!(system.rn, V)
    st = ModelingToolkit.states(system.rn)
    perm = circshift(collect(1:length(st)), 1)
    Catalyst.reorder_states!(system.rn, perm)

    sds = PathWeightSampling.SDEDrivenSystem(
        sn, system.rn, system.xn,
        vcat(0.0, system.u0[1], system.u0[2:end]), # u0
        ps, system.pr, system.px,
        system.dtimes,
        aggregator=get(kwargs, :aggregator, SortingDirect()),
        dist_aggregator=get(kwargs, :dist_aggregator, DepGraphDirect()),
    )
end

struct ChemotaxisJumps <: AbstractJumpSet
    KD_a::Float64
    KD_i::Float64
    N::Int64
    k_B::Float64
    k_R::Float64
    k_A::Float64
    k_Z::Float64
    δf::Float64
    m_0::Float64
    ligand::Int32
    receptors::UnitRange{Int32}
    Yp::Int32
    Y::Int32
end

num_species(jumps::ChemotaxisJumps) = 3 + length(jumps.receptors)
num_reactions(jumps::ChemotaxisJumps) = 3 * length(jumps.receptors) + 1

@inline function reaction_type(j::ChemotaxisJumps, rxidx::Integer)
    nstates = length(j.receptors)
    type, m = divrem(rxidx - 1, nstates)
    (type, m)
end

struct ChemotaxisCache
    c_prev::Ref{Float64}
    z_m::Vector{Float64}
    p_a::Vector{Float64}
end

function Base.copy(cache::ChemotaxisCache)
    ChemotaxisCache(Ref(cache.c_prev[]), copy(cache.z_m), copy(cache.p_a))
end

function initialize_cache(jumps::ChemotaxisJumps)
    E_m = [jumps.δf * (m - jumps.m_0) for m in 0:length(jumps.receptors)-1]
    ChemotaxisCache(
        Ref(0.0),
        exp.(-E_m),
        fill(1 / 2, length(jumps.receptors))
    )
end

function update_cache!(agg, jumps::ChemotaxisJumps)
    ligand_c = agg.u[jumps.ligand]
    if agg.cache.c_prev[] != ligand_c
        z_a = (1 + (ligand_c / jumps.KD_a))^jumps.N
        z_i = (1 + (ligand_c / jumps.KD_i))^jumps.N
        @. agg.cache.p_a = z_a * agg.cache.z_m / (z_a * agg.cache.z_m + z_i)
        agg.cache.c_prev[] = ligand_c
    end
end

@inline @fastmath function evalrxrate(agg::AbstractJumpRateAggregator, rxidx::Int64, jumps::ChemotaxisJumps)
    nstates = length(jumps.receptors)
    type, m = reaction_type(jumps, rxidx)

    speciesvec = agg.u
    @inbounds p_a = agg.cache.p_a[m+1]

    @inbounds rec_m = jumps.receptors[m+1]

    if type == 0 # methylate
        if m == (nstates - 1)
            return 0.0
        end
        @inbounds (1 - p_a) * jumps.k_R * speciesvec[rec_m]
    elseif type == 1 # demethylate
        if m == 0
            return 0.0
        end
        @inbounds p_a * jumps.k_B * speciesvec[rec_m]
    elseif type == 2 # phosphorylate
        @inbounds p_a * jumps.k_A * speciesvec[rec_m] * speciesvec[jumps.Y]
    else # dephosphorylate
        @inbounds jumps.k_Z * speciesvec[jumps.Yp]
    end
end

function executerx!(speciesvec::AbstractVector, rxidx::Integer, jumps::ChemotaxisJumps)
    type, m = reaction_type(jumps, rxidx)

    if type == 0 # methylate
        @inbounds speciesvec[jumps.receptors[m+1]] -= 1
        @inbounds speciesvec[jumps.receptors[m+2]] += 1
    elseif type == 1 # demethylate
        @inbounds speciesvec[jumps.receptors[m+1]] -= 1
        @inbounds speciesvec[jumps.receptors[m]] += 1
    elseif type == 2 # phosphorylate
        @inbounds speciesvec[jumps.Y] -= 1
        @inbounds speciesvec[jumps.Yp] += 1
    else # dephosphorylate
        @inbounds speciesvec[jumps.Y] += 1
        @inbounds speciesvec[jumps.Yp] -= 1
    end
end

function dependend_species(jumps::ChemotaxisJumps, rxidx::Integer)
    type, m = reaction_type(jumps, rxidx)

    if type == 0 # methylate
        [jumps.receptors[m+1]]
    elseif type == 1 # demethylate
        [jumps.receptors[m+1]]
    elseif type == 2 # phosphorylate
        [jumps.receptors[m+1], jumps.Y]
    else # dephosphorylate
        [jumps.Yp]
    end
end

function mutated_species(jumps::ChemotaxisJumps, rxidx::Integer)
    type, m = reaction_type(jumps, rxidx)
    nstates = length(jumps.receptors)

    if type == 0 # methylate
        if m == (nstates - 1)
            return []
        end
        [jumps.receptors[m+1], jumps.receptors[m+2]]
    elseif type == 1 # demethylate
        if m == 0
            return []
        end
        [jumps.receptors[m+1], jumps.receptors[m]]
    elseif type == 2 # phosphorylate
        [jumps.Y, jumps.Yp]
    else # dephosphorylate
        [jumps.Y, jumps.Yp]
    end
end

function simple_chemotaxis_system(;
    n_clusters=25,
    methylation_sites=4,
    duration=200.0,
    dt=0.1,
    sde_dt=dt / 10,
    c_0=100.0,
    Kₐ=2900.0,
    Kᵢ=18.0,
    n=15, # cooperativity
    k_R=0.1,
    k_B=0.2,
    a_0=k_R / (k_R + k_B),
    δf=-2.0,
    m_0=0.5,
    k_Z=10.0,
    phi_y=1 / 6,
    k_A=k_Z * phi_y / ((1 - phi_y) * a_0 * n_clusters),
    velocity_decay=0.862,
    velocity_noise=sqrt(2 * velocity_decay * 157.1),
    gradient_steepness=0.2e-3,
    harmonic_rate=nothing
)
    jumps = ChemotaxisJumps(
        Kₐ,
        Kᵢ,
        n,
        k_B,
        k_R,
        k_A,
        k_Z,
        δf,
        m_0,
        1,
        range(2, length=n * methylation_sites + 1),
        n * methylation_sites + 3,
        n * methylation_sites + 4
    )

    m = 0:(n*methylation_sites)
    a_m = activity_given_methylation.(m, c=c_0, N=n, K_a=Kₐ, K_i=Kᵢ, δfₘ=δf, m₀=m_0)
    p_m = steady_state_methylation(a_m, k_B=k_B, k_R=k_R)

    u0 = zeros(Float64, num_species(jumps))
    u0[jumps.ligand] = c_0
    for s in systematic_sample(p_m, N=n_clusters)
        u0[jumps.receptors[s]] += 1
    end
    u0[jumps.Y] = 10000

    rid_to_gid = zeros(Int32, num_reactions(jumps))
    rid_to_gid[length(jumps.receptors)*2+1:length(jumps.receptors)*3] .= 1
    rid_to_gid[end] = 2
    rid_to_gid

    ModelingToolkit.@variables t V(t) L(t)
    ModelingToolkit.@parameters λ σ g ω₀

    D = ModelingToolkit.Differential(t)

    if harmonic_rate === nothing
        eqs = [
            D(V) ~ -λ * V,
            D(L) ~ g * L * V
        ]
        noiseeqs = [σ, 0]
        sparams = [λ, σ, g]
        ps = [velocity_decay, velocity_noise, gradient_steepness]
    else
        c₀ = 100.0 # μM
        eqs = [
            D(V) ~ -ω₀^2 * (L - c₀) - λ * V,
            D(L) ~ g * c₀ * V
        ]
        noiseeqs = [σ, 0]
        sparams = [λ, σ, g, ω₀]
        ps = [velocity_decay, velocity_noise, gradient_steepness, harmonic_rate]
    end

    tspan = (0.0, duration)
    ModelingToolkit.@named sn = SDESystem(eqs, noiseeqs, t, [V, L], sparams)
    s_prob = SDEProblem(sn, [0.0, u0[1]], tspan, ps)

    traced_reactions = BitSet(length(jumps.receptors)*2+1:length(jumps.receptors)*3+1)
    @show traced_reactions
    @assert length(traced_reactions) == length(jumps.receptors) + 1
    @assert maximum(traced_reactions) == num_reactions(jumps)

    HybridJumpSystem(
        DepGraphDirect(),
        jumps,
        u0,
        tspan,
        dt,
        s_prob,
        sde_dt,
        rid_to_gid,
        traced_reactions
    )
end

getname(sym) = String(ModelingToolkit.operation(sym).name)

function parse_receptor(species)
    sname = getname(species)
    mtch = match(r"([A-Z])_(\d+)_(\d+)", sname)
    if mtch !== nothing
        l = parse(Int, mtch.captures[2])
        m = parse(Int, mtch.captures[3])
        (l, m)
    else
        nothing
    end
end

function receptor_states(rs::ReactionSystem)
    smap = Catalyst.speciesmap(rs)
    xf = KeepSomething() do (species, index)
        result = parse_receptor(species)
        if result === nothing
            nothing
        else
            result => index
        end
    end
    smap |> xf
end

receptor_states(system::ComplexSystem) = receptor_states(reaction_network(system))

function active_receptors(conf::SRXconfiguration, system::ComplexSystem)
    rstates = receptor_states(system)

    E0 = system.pr[1]
    δg = system.pr[2]
    δf = system.pr[3]

    p_active(l, m) = 1 / (1 + exp(E0 + l * δg + m * δf))

    p_a = rstates |> Map(((l, m), i)::Pair -> i => p_active(l, m)) |> collect

    f = function (u)
        sum(p_a) do (i, p)
            p * u[i]
        end
    end

    merge_trajectories(conf.s_traj, conf.r_traj, conf.x_traj) |> Map((u, t, i)::Tuple -> (ensurevec(f(u)), t, i)) |> collect_trajectory
end

# function mwc_params(;
#     ϕ_y,
#     N=10, # cooperativity
#     n_clusters=100, # number of receptor clusters
#     tau_l=1.0,
#     K_i=18,
#     K_a=2900,
#     δf=-2.0,
#     m0=0.5 * N,
#     a0=0.3,
#     k_B=0.15,
#     k_R=0.075,
#     k_Z=10.0,
#     k_A=k_Z * ϕ_y / (a0 * n_clusters), velocity_decay=0.862,
#     velocity_noise=sqrt(2 * velocity_decay * 157.1),
#     gradient_steepness=0.2e-3
# )
#     ps = [velocity_decay, velocity_noise, gradient_steepness]
#     pr = [N, m0, δf, K_i, K_a, k_B, k_R]
#     px = [k_A, k_Z]
#     ps, pr, px
# end

# function mwc_chemotaxis_system(;
#     mean_l=100, # mean ligand concentration [μM]
#     n_clusters=800, # number of receptor clusters
#     N=10, # cooperativity
#     M=4 * N, # methylation sites
#     n_chey=10_000, # number of total cheY
#     ϕ_y=1.0 / 6.0, # fraction of phosphorylated CheY
#     dtimes=0:0.1:20.0,
#     aggregator=JumpProcesses.SortingDirect(),
#     dist_aggregator=DepGraphDirect(),
#     params...
# )
#     ps, pr, px = mwc_params(; N, ϕ_y, n_clusters, params...)

#     ModelingToolkit.@variables t V(t) L(t)
#     ModelingToolkit.@parameters λ σ g

#     D = ModelingToolkit.Differential(t)

#     eqs = [
#         D(V) ~ -λ * V,
#         D(L) ~ g * L * V
#     ]
#     noiseeqs = [σ, 0]
#     sparams = [λ, σ, g]

#     ModelingToolkit.@named sn = SDESystem(eqs, noiseeqs, t, [V, L], sparams)

#     rn = Catalyst.make_empty_network()

#     Catalyst.@parameters t N m0 δf K_i K_a k_B k_R ρ
#     Catalyst.@variables Y(t) Yp(t)

#     # add input species first
#     Catalyst.addspecies!(rn, V)
#     Catalyst.addspecies!(rn, L)

#     Catalyst.addparam!(rn, N)
#     Catalyst.addparam!(rn, m0)
#     Catalyst.addparam!(rn, δf)
#     Catalyst.addparam!(rn, K_i)
#     Catalyst.addparam!(rn, K_a)
#     Catalyst.addparam!(rn, k_B)
#     Catalyst.addparam!(rn, k_R)

#     xn = @reaction_network begin
#         μ, Yp --> Y
#     end μ

#     Catalyst.addparam!(xn, ρ)

#     spmap = Dict()
#     for m = 0:M
#         receptor_species = ModelingToolkit.Num(ModelingToolkit.variable(Symbol("R_", m), T=FnType{Tuple{Any},Real}))(t)

#         spmap[m] = receptor_species

#         Catalyst.addspecies!(xn, receptor_species)
#         Catalyst.addspecies!(rn, receptor_species)
#     end

#     p_active(c, m) = m == 0 ? 0.0 : (m == M ? 1.0 : 1 / (1 + exp(N * log((1 + c / K_i) / (1 + c / K_a)) + δf * (m - m0))))

#     for m = 0:M
#         if m > 0
#             demethylation_rate = p_active(L, m) * k_B
#             demethylate_active = @reaction $demethylation_rate, $(spmap[m]) --> $(spmap[m-1])
#             methylation_rate = (1 - p_active(L, m)) * k_R
#             methylate_inactive = @reaction $methylation_rate, $(spmap[m-1]) --> $(spmap[m])

#             Catalyst.addreaction!(rn, demethylate_active)
#             Catalyst.addreaction!(rn, methylate_inactive)
#         end


#         # every receptor phosphorylates Y with rate ρ if the receptor is active
#         # if the receptor is inactive, no phosphorylation can happen
#         receptor = spmap[m]
#         phosphorylation_rate = p_active(L, m) * ρ
#         phosphorylation = @reaction $phosphorylation_rate, $receptor + Y --> $receptor + Yp
#         Catalyst.addreaction!(xn, phosphorylation)
#     end

#     joint = ModelingToolkit.extend(xn, rn)

#     spec2index = Catalyst.speciesmap(joint)
#     u0 = zeros(Float64, length(spec2index))
#     u0[2] = mean_l
#     u0[spec2index[Yp]] = round(n_chey * ϕ_y)
#     u0[spec2index[Y]] = n_chey - u0[spec2index[Yp]]

#     # here we approximate the steady states of the individual receptors
#     # to choose a sensible initial condition
#     N = pr[1]
#     m0 = pr[2]
#     δf = pr[3]
#     K_i = pr[4]
#     K_a = pr[5]
#     Z_a = (1 + mean_l / K_a)
#     Z_i = (1 + mean_l / K_i)
#     a = @. 1 / (1 + exp(N * log(Z_i / Z_a) + δf * ((0:M) - m0)))
#     p_m = steady_state_methylation(a, k_B=pr[6], k_R=pr[7])
#     for state in systematic_sample(p_m; N=n_clusters)
#         m = state - 1
#         u0[spec2index[spmap[m]]] += 1
#     end

#     PathWeightSampling.SDEDrivenSystem(
#         sn, rn, xn,
#         u0,
#         ps, pr, px,
#         dtimes,
#         aggregator=get(params, :aggregator, SortingDirect()),
#         dist_aggregator=get(params, :dist_aggregator, DepGraphDirect()),
#     )
# end

precompile(gene_expression_system, ())
precompile(chemotaxis_system, ())
precompile(cooperative_chemotaxis_system, ())
precompile(sde_chemotaxis_system, ())
