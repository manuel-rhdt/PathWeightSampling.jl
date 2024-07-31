import .SMC: systematic_sample

using .SSA
using LinearAlgebra
using StochasticDiffEq
using StaticArrays
using Accessors

import Random

"""
    gene_expression_system(;
        mean_s=50,
        mean_x=mean_s,
        corr_time_s=1.0,
        corr_time_x=0.1,
        kappa=mean_s / corr_time_s,
        lambda=1 / corr_time_s,
        mu=1 / corr_time_x,
        rho=mu * mean_x / mean_s,
        u0=[mean_s, mean_x],
        dtimes=0:0.1:2.0
    )

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
import PathWeightSampling as PWS
system = PWS.gene_expression_system(kappa = 10.0, lambda = 0.1, rho = 1.0, mu = 1.0)

# output

MarkovJumpSystem with 2 species and 4 reactions
k = 10.00: ∅ ---> S
k = 0.10: S ---> ∅
k = 1.00: S ---> S + X
k = 1.00: X ---> ∅

Initial condition:
    S = 50
    X = 50
```

Alternatively, the reaction rates can be specified indirectly through the following arguments:
- `mean_s`: The average copy number of S when the system has relaxed to steady state.
- `mean_x`: The average copy number of X when the system has relaxed to steady state.
- `corr_time_s`: The input signal correlation time. The shorter this time is, the faster the fluctuations in the input signal.
- `corr_time_x`: The output signal correlation time. This sets the timescale of output fluctuations.

```jldoctest
import PathWeightSampling as PWS
system = PWS.gene_expression_system(mean_s=25, mean_x=50, corr_time_s=1.0, corr_time_x=0.75)

# output

MarkovJumpSystem with 2 species and 4 reactions
k = 25.00: ∅ ---> S
k = 1.00: S ---> ∅
k = 2.67: S ---> S + X
k = 1.33: X ---> ∅

Initial condition:
    S = 25
    X = 50
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
    u0=[mean_s, mean_x],
    dtimes=0:0.1:2.0
)
    species = [:S, :X]
    rates = SA[kappa, lambda, rho, mu]
    rstoich = (SA{Pair{Int, Int}}[], SA[1 => 1], SA[1 => 1], SA[2 => 1])
    nstoich = (SA[1 => 1], SA[1 => -1], SA[2 => 1], SA[2 => -1])

    reactions = ReactionSet(rates, rstoich, nstoich, species)

    MarkovJumpSystem(
        GillespieDirect(),
        reactions,
        u0,
        extrema(dtimes),
        :S,
        :X
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

SSA.num_species(jumps::ChemotaxisJumps) = 3 + length(jumps.receptors)
SSA.num_reactions(jumps::ChemotaxisJumps) = 3 * length(jumps.receptors) + 1
function SSA.speciesnames(jumps::ChemotaxisJumps)
    vcat(
        :L,
        [Symbol(:R, i) for i in 0:(length(jumps.receptors) - 1)],
        :Yp,
        :Y
    )
end

@inline function reaction_type(j::ChemotaxisJumps, rxidx::Integer)
    nstates = length(j.receptors)
    type, m = divrem(rxidx - 1, nstates)
    (type, m)
end

struct ChemotaxisCache
    c_prev::Float64
    z_m::Vector{Float64}
    p_a::Vector{Float64}
end

function Base.copy(cache::ChemotaxisCache)
    ChemotaxisCache(cache.c_prev, copy(cache.z_m), copy(cache.p_a))
end

function SSA.initialize_cache(jumps::ChemotaxisJumps)
    E_m = [jumps.δf * (m - jumps.m_0) for m in 0:length(jumps.receptors)-1]
    ChemotaxisCache(
        0.0,
        exp.(-E_m),
        fill(1 / 2, length(jumps.receptors))
    )
end

function SSA.update_cache(agg, jumps::ChemotaxisJumps)
    ligand_c = agg.u[jumps.ligand]
    if agg.cache.c_prev != ligand_c
        z_a = (1 + (ligand_c / jumps.KD_a))^jumps.N
        z_i = (1 + (ligand_c / jumps.KD_i))^jumps.N
        @. agg.cache.p_a = z_a * agg.cache.z_m / (z_a * agg.cache.z_m + z_i)
        @set agg.cache.c_prev = ligand_c
    else
        agg
    end
end

function SSA.make_reaction_groups(jumps::ChemotaxisJumps, species::Symbol)
    rid_to_gid = zeros(Int32, num_reactions(jumps))
    rid_to_gid[length(jumps.receptors)*2+1:length(jumps.receptors)*3] .= 1
    rid_to_gid[end] = 2
    rid_to_gid
end

@inline function SSA.evalrxrate(agg::AbstractJumpRateAggregator, rxidx::Int64, jumps::ChemotaxisJumps)
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

function SSA.executerx!(speciesvec::AbstractVector, rxidx::Integer, jumps::ChemotaxisJumps)
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

function SSA.dependend_species(jumps::ChemotaxisJumps, rxidx::Integer)
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

function SSA.mutated_species(jumps::ChemotaxisJumps, rxidx::Integer)
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

function average_activity(jumps::ChemotaxisJumps, agg::AbstractJumpRateAggregator)
    n_receptors = sum(agg.u[jumps.receptors])
    sum(agg.cache.p_a .* agg.u[jumps.receptors]) / n_receptors
end

function max_methylation_level(jumps::ChemotaxisJumps, u::AbstractVector)
    (length(jumps.receptors) - 1) * Int(sum(@view u[jumps.receptors]))
end

function methylation_level(jumps::ChemotaxisJumps, u::AbstractVector)
    sum = 0
    for (i, m) in zip(jumps.receptors, 0:length(jumps.receptors)-1)
        sum += Int(u[i]) * m
    end
    sum
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

"""

    chemotaxis_system(;
        n_clusters=25,
        n_chey=10000,
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
        m_0=0.5 * n,
        k_Z=10.0,
        phi_y=1 / 6,
        k_A=k_Z * phi_y / ((1 - phi_y) * a_0 * n_clusters),
        velocity_decay=0.862,
        velocity_noise=sqrt(2 * velocity_decay * 157.1),
        gradient_steepness=0.2e-3,
    )

Create a system for a stochastic chemotaxis model.

# Model description

This model describes the bacterial chemotaxis signaling network.

    Rml -> Rm(l+1), with rate kon L(t) (lmax - l): ligand binding to active state
    Rm(l+1) -> Rml, with rate koff_A l: ligand unbinding
    Rml -> R(m+1)l with rate k_R: methylation rate
    R(m+1)l -> Rml with rate k_B: demethylation rate

# Example

```jldoctest
import PathWeightSampling as PWS
system = PWS.chemotaxis_system()

# output

HybridJumpSystem with 64 species and 184 reactions
PathWeightSampling.ChemotaxisJumps(2900.0, 18.0, 15, 0.2, 0.1, 0.23999999999999996, 10.0, -2.0, 7.5, 1, 2:62, 63, 64)

Initial condition:
    L = 100.0
    R0 = 0.0
    R1 = 0.0
    R2 = 0.0
    R3 = 0.0
    R4 = 0.0
    R5 = 0.0
    R6 = 0.0
    R7 = 0.0
    R8 = 0.0
    R9 = 0.0
    R10 = 0.0
    R11 = 0.0
    R12 = 0.0
    R13 = 0.0
    R14 = 0.0
    R15 = 0.0
    R16 = 0.0
    R17 = 0.0
    R18 = 0.0
    R19 = 1.0
    R20 = 8.0
    R21 = 11.0
    R22 = 5.0
    R23 = 0.0
    R24 = 0.0
    R25 = 0.0
    R26 = 0.0
    R27 = 0.0
    R28 = 0.0
    R29 = 0.0
    R30 = 0.0
    R31 = 0.0
    R32 = 0.0
    R33 = 0.0
    R34 = 0.0
    R35 = 0.0
    R36 = 0.0
    R37 = 0.0
    R38 = 0.0
    R39 = 0.0
    R40 = 0.0
    R41 = 0.0
    R42 = 0.0
    R43 = 0.0
    R44 = 0.0
    R45 = 0.0
    R46 = 0.0
    R47 = 0.0
    R48 = 0.0
    R49 = 0.0
    R50 = 0.0
    R51 = 0.0
    R52 = 0.0
    R53 = 0.0
    R54 = 0.0
    R55 = 0.0
    R56 = 0.0
    R57 = 0.0
    R58 = 0.0
    R59 = 0.0
    R60 = 0.0
    Yp = 0.0
    Y = 10000.0
```
"""
function chemotaxis_system(;
    n_clusters=25,
    n_chey=10000,
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
    m_0=0.5 * n,
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

    # we generate initial conditions close to the steady state
    u0 = zeros(Float64, num_species(jumps))
    u0[jumps.ligand] = c_0
    for s in systematic_sample(Random.Xoshiro(1), p_m, N=n_clusters)
        u0[jumps.receptors[s]] += 1
    end
    u0[jumps.Y] = n_chey

    # if harmonic_rate === nothing
    function det_evolution!(du, u, p, t)
        λ, σ, g = p
        du[1] = -λ * u[1]
        du[2] = g * u[1] * u[2]
    end

    function noise!(du, u, p, t)
        λ, σ, g = p
        du[1] = σ
        du[2] = 0
    end
    ps = [velocity_decay, velocity_noise, gradient_steepness]
    # else
    #     c₀ = 100.0 # μM
    #     eqs = [
    #         D(V) ~ -ω₀^2 * (L - c₀) - λ * V,
    #         D(L) ~ g * c₀ * V
    #     ]
    #     noiseeqs = [σ, 0]
    #     sparams = [λ, σ, g, ω₀]
    #     ps = [velocity_decay, velocity_noise, gradient_steepness, harmonic_rate]
    # end

    tspan = (0.0, duration)
    s_prob = SDEProblem(det_evolution!, noise!, [0.0, u0[1]], tspan, ps)
    sde_species_mapping = [2 => 1]

    HybridJumpSystem(
        DepGraphDirect(),
        jumps,
        u0,
        tspan,
        dt,
        s_prob,
        sde_dt,
        :L,
        :Yp,
        sde_species_mapping
    )
end

