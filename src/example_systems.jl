import Catalyst:@reaction_network

function gene_expression_system(; mean_s=50, corr_time_s=1.0, corr_time_x=0.1, dtimes=0:0.1:2.0)
    sn = @reaction_network begin
        κ, ∅ --> S
        λ, S --> ∅
    end κ λ

    xn = @reaction_network begin
        ρ, S --> X + S
        μ, X --> ∅ 
    end ρ μ

    λ = 1 / corr_time_s
    κ = mean_s * λ
    μ = 1 / corr_time_x
    ρ = μ
    mean_x = mean_s

    u0 = SA[mean_s, mean_x]
    ps = [κ, λ]
    px = [ρ, μ]

    SXsystem(sn, xn, u0, ps, px, dtimes)
end

function chemotaxis_system(; mean_L=20, num_receptors=10000, Y_tot=5000, L_timescale=1.0, LR_timescale=0.01, LR_ratio=0.5, Y_timescale=0.1, Y_ratio=1/6, q=0, dtimes=0:0.1:2.0)
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

    SRXsystem(sn, rn, xn, u0, ps, pr, px, dtimes)
end

import Catalyst
import ModelingToolkit

function cooperative_chemotaxis_system(;
    lmax = 3,
    mmax = 9,

    K_on = 500,
    K_off = 25,

    δf = -3.0,

    γ = 0.06,

    mean_l = 50,

    phosphorylate = 0.1,
    dephosphorylate = 0.1,

    n_clusters = 100,
    n_chey = 10_000,
    dtimes = 0:0.1:20.0
)
    delta_f = δf

    sn = @reaction_network begin
        κ, ∅ --> L
        λ, L --> ∅
    end κ λ

    rn = Catalyst.make_empty_network()

    @Catalyst.parameters t E0 τ0 δg δf lba lbi lda ldi mda mbi ρ
    @Catalyst.variables L(t) Y(t) Yp(t)

    Catalyst.addspecies!(rn, L)

    Catalyst.addparam!(rn, E0)
    Catalyst.addparam!(rn, τ0)
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
    for l=0:lmax, m=0:mmax
        # active_species = ModelingToolkit.Num(ModelingToolkit.Variable{ModelingToolkit.FnType{Tuple{Any},Real}}(Symbol("A_", l, "_", m)))(t)
        # inactive_species = ModelingToolkit.Num(ModelingToolkit.Variable{ModelingToolkit.FnType{Tuple{Any},Real}}(Symbol("I_", l, "_", m)))(t)
        receptor_species = ModelingToolkit.Num(ModelingToolkit.Variable{ModelingToolkit.FnType{Tuple{Any},Real}}(Symbol("R_", l, "_", m)))(t)

        # spmap[(l, m, :active)] = active_species
        # spmap[(l, m, :inactive)] = inactive_species
        spmap[(l, m)] = receptor_species

        # Catalyst.addspecies!(rn, active_species)
        # Catalyst.addspecies!(xn, active_species)
        # Catalyst.addspecies!(rn, inactive_species)
        Catalyst.addspecies!(xn, receptor_species)
        Catalyst.addspecies!(rn, receptor_species)
    end

    γ_B = γ / (3 * lmax * abs(delta_f))
    γ_R = γ_B / 2

    for l=0:lmax, m=0:mmax
        p_active = 1 / (1 + exp(E0 + l*δg + m*δf))

        if l > 0
            ligand_bind = Catalyst.Reaction((lba * p_active + lbi * (1 - p_active)) * (lmax + 1 - l), [spmap[(l-1, m)], L], [spmap[(l, m)], L])
            ligand_unbind = Catalyst.Reaction((lda * p_active + ldi * (1 - p_active)) * l, [spmap[(l, m)]], [spmap[(l-1, m)]])
            Catalyst.addreaction!(rn, ligand_bind)
            Catalyst.addreaction!(rn, ligand_unbind)
        end

        if m > 0
            # methylate_active = Catalyst.Reaction(methyl_rate * (mmax + 1 - m), [spmap[(l, m-1, :active)]], [spmap[(l, m, :active)]])
            demethylate_active = Catalyst.Reaction(mda * p_active, [spmap[(l, m)]], [spmap[(l, m-1)]])
            methylate_inactive = Catalyst.Reaction(mbi * (1 - p_active), [spmap[(l, m-1)]], [spmap[(l, m)]])
            # demethylate_inactive = Catalyst.Reaction(mda, [spmap[(l, m, :inactive)]], [spmap[(l, m-1, :inactive)]])

            # Catalyst.addreaction!(rn, methylate_active)
            Catalyst.addreaction!(rn, demethylate_active)
            Catalyst.addreaction!(rn, methylate_inactive)
            # Catalyst.addreaction!(rn, demethylate_inactive)
        end


        # every receptor phosphorylates Y with rate ρ if the receptor is active
        # if the receptor is inactive, no phosphorylation can happen
        receptor = spmap[(l, m)]
        phosphorylation = Catalyst.Reaction(ρ * p_active, [Y, receptor], [Yp, receptor])
        Catalyst.addreaction!(xn, phosphorylation)
    end

    joint = merge(merge(sn, rn), xn)

    u0 = zeros(Int, length(spmap) + 3)
    u0[1] = round(Int, mean_l)
    u0[Catalyst.speciesmap(joint)[spmap[(0, 0)]]] = n_clusters
    u0[Catalyst.speciesmap(joint)[Y]] = n_chey

    ps = [mean_l, 1.0]

    lr = 0.05
    pr = [0.0, 0.2, log(K_on/K_off), delta_f, lr, lr, lr * K_on, lr * K_off, γ_B, γ_R]
    px = [dephosphorylate, phosphorylate]

    SRXsystem(sn, rn, xn, u0, ps, pr, px, dtimes; aggregator=DiffEqJump.DirectCR())
end

function active_indices(rs, firstletter = "A")
    smap = Catalyst.speciesmap(rs)
    result = Int[]
    for (species, index) in smap
        sname = String(ModelingToolkit.operation(species).name)
        if startswith(sname, firstletter)
            push!(result, index)
        end
    end

    result
end

# using Plots
# begin
# system = cooperative_chemotaxis_system(;delta_f=-1, mean_s=5)
# joint_n = reaction_network(system)
# @Catalyst.parameters t
# @Catalyst.variables Yp(t)
# act_i = active_indices(joint_n)
# inact_i = active_indices(joint_n, "I")
# yp_i = Catalyst.speciesmap(joint_n)[Yp]
# @time res = _solve(system)
# act_sum = vec(sum(res[act_i, :], dims=1))
# inact_sum = vec(sum(res[inact_i, :], dims=1))
# plot(res.t, [inact_sum, res[1,:], res[yp_i,:]./100])
# end
