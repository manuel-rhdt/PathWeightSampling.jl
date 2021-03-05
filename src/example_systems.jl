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

function chemotaxis_system(; mean_L=20, num_receptors=10, Y_tot=50, L_timescale=1.0, LR_timescale=0.5, LR_ratio=0.5, Y_timescale=0.1, Y_ratio=1/6, q=0, dtimes=0:0.1:2.0)
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

    u0 = [mean_L, mean_R, mean_LR, mean_Y, mean_Yp]
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
    lmax = 4,
    mmax = 9,

    k_on = 5,
    k_off = 5,
    km = 5,
    kdm = 5,

    delta_e = -1,
    delta_g = 2,
    delta_f = -1,

    mean_s = 50
)
    sn = @reaction_network begin
        κ, ∅ --> L
        λ, L --> ∅
    end κ λ

    rn = Catalyst.make_empty_network()

    @Catalyst.parameters t
    @Catalyst.variables L(t) Y(t) Yp(t)

    Catalyst.addspecies!(rn, L)

    xn = @reaction_network begin
        χ, Yp --> Y
    end χ

    spmap = Dict()
    for l=0:lmax, m=0:mmax
        active_species = ModelingToolkit.Num(ModelingToolkit.Variable{ModelingToolkit.FnType{Tuple{Any},Real}}(Symbol("A_", l, "_", m)))(t)
        inactive_species = ModelingToolkit.Num(ModelingToolkit.Variable{ModelingToolkit.FnType{Tuple{Any},Real}}(Symbol("I_", l, "_", m)))(t)

        spmap[(l, m, :active)] = active_species
        spmap[(l, m, :inactive)] = inactive_species

        Catalyst.addspecies!(rn, active_species)
        Catalyst.addspecies!(xn, active_species)
        Catalyst.addspecies!(rn, inactive_species)
    end

    for l=0:lmax, m=0:mmax
        if l > 0
            ligand_bind_active = Catalyst.Reaction(k_on * (lmax + 1 - l), [spmap[(l-1, m, :active)], L], [spmap[(l, m, :active)], L])
            ligand_unbind_active = Catalyst.Reaction(k_off * l, [spmap[(l, m, :active)]], [spmap[(l-1, m, :active)]])
            ligand_bind_inactive = Catalyst.Reaction(k_on * (lmax + 1 - l), [spmap[(l-1, m, :inactive)], L], [spmap[(l, m, :inactive)], L])
            ligand_unbind_inactive = Catalyst.Reaction(k_off * l, [spmap[(l, m, :inactive)]], [spmap[(l-1, m, :inactive)]])

            Catalyst.addreaction!(rn, ligand_bind_active)
            Catalyst.addreaction!(rn, ligand_unbind_active)
            Catalyst.addreaction!(rn, ligand_bind_inactive)
            Catalyst.addreaction!(rn, ligand_unbind_inactive)
        end

        if m > 0
            methylate_active = Catalyst.Reaction(km * (mmax + 1 - m), [spmap[(l, m-1, :active)]], [spmap[(l, m, :active)]])
            demethylate_active = Catalyst.Reaction(kdm * m, [spmap[(l, m, :active)]], [spmap[(l, m-1, :active)]])
            methylate_inactive = Catalyst.Reaction(km * (mmax + 1 - m), [spmap[(l, m-1, :inactive)]], [spmap[(l, m, :inactive)]])
            demethylate_inactive = Catalyst.Reaction(kdm * m, [spmap[(l, m, :inactive)]], [spmap[(l, m-1, :inactive)]])

            Catalyst.addreaction!(rn, methylate_active)
            Catalyst.addreaction!(rn, demethylate_active)
            Catalyst.addreaction!(rn, methylate_inactive)
            Catalyst.addreaction!(rn, demethylate_inactive)
        end

        switch_active = Catalyst.Reaction(exp(-delta_e-l*delta_g-m*delta_f), [spmap[(l, m, :inactive)]], [spmap[(l, m, :active)]])
        switch_inactive = Catalyst.Reaction(1, [spmap[(l, m, :active)]], [spmap[(l, m, :inactive)]])
        Catalyst.addreaction!(rn, switch_active)
        Catalyst.addreaction!(rn, switch_inactive)

        active_spec = spmap[(l, m, :inactive)]
        phosphorylate = Catalyst.Reaction(5e-2, [Y, active_spec], [Yp, active_spec])
        Catalyst.addreaction!(xn, phosphorylate)
    end

    joint = merge(merge(sn, rn), xn)

    u0 = zeros(Int, length(spmap) + 3)
    u0[1] = round(Int, mean_s)
    u0[Catalyst.speciesmap(joint)[spmap[(0, 0, :inactive)]]] = 100
    u0[Catalyst.speciesmap(joint)[Y]] = 10000

    ps = [mean_s, 1.0]
    pr = []
    px = [2.0]
    dtimes = 0:0.1:20.0

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
