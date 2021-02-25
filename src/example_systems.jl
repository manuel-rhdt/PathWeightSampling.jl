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

function chemotaxis_system(; mean_L=20, num_receptors=10, Y_tot=50, L_timescale=1.0, LR_timescale=0.5, LR_ratio=0.5, Y_timescale=0.1, Y_ratio=0.5, dtimes=0:0.1:2.0)
    mean_LR = num_receptors * LR_ratio
    mean_R = num_receptors - mean_LR

    mean_Yp = Y_tot * Y_ratio
    mean_Y = Y_tot - mean_Yp

    sn = @reaction_network begin
        κ, ∅ --> L
        λ, L --> ∅
    end κ λ

    rn = @reaction_network begin
        ρ, L + R --> L + LR
        μ, LR --> R
    end ρ μ

    xn = @reaction_network begin
        (δ, δb), LR + Y ↔ Yp + LR
        (χ, χb), Yp ↔ Y
    end δ δb χ χb

    u0 = SA[mean_L, mean_R, mean_LR, mean_Y, mean_Yp]
    ps = [mean_L, 1 / L_timescale]
    pr = [mean_LR / (LR_timescale * mean_R * mean_L), 1 / LR_timescale]
    px = [mean_Yp / (Y_timescale * mean_Y * mean_LR), 1e-3, 1 / Y_timescale, 1e-3]

    SRXsystem(sn, rn, xn, u0, ps, pr, px, dtimes)
end

import Catalyst
using ModelingToolkit

function cooperative_chemotaxis_system()
    lmax = 2
    mmax = 3

    k_on = 5
    k_off = 5
    km = 5
    kdm = 5

    sn = @reaction_network begin
        κ, ∅ --> L
        λ, L --> ∅
    end κ λ

    rn = Catalyst.make_empty_network()

    @Catalyst.parameters t
    @Catalyst.variables L(t) Y(t) Yp(t)

    Catalyst.addspecies!(rn, L)

    xn = @reaction_network begin
        (χ, χb), Yp ↔ Y
    end χ χb

    spmap = Dict()
    for l=0:lmax, m=0:mmax
        active_species = Num(Variable{ModelingToolkit.FnType{Tuple{Any},Real}}(Symbol("A_", l, "_", m)))(t)
        inactive_species = Num(Variable{ModelingToolkit.FnType{Tuple{Any},Real}}(Symbol("I_", l, "_", m)))(t)

        spmap[(l, m, :active)] = active_species
        spmap[(l, m, :inactive)] = inactive_species

        Catalyst.addspecies!(rn, active_species)
        Catalyst.addspecies!(xn, active_species)
        Catalyst.addspecies!(rn, inactive_species)
    end

    for l=0:lmax, m=0:mmax
        if l > 0
            ligand_bind_active = Catalyst.Reaction(k_on * L * (lmax + 1 - l), [spmap[(l-1, m, :active)]], [spmap[(l, m, :active)]])
            ligand_unbind_active = Catalyst.Reaction(k_off * l, [spmap[(l, m, :active)]], [spmap[(l-1, m, :active)]])
            ligand_bind_inactive = Catalyst.Reaction(k_on * L * (lmax + 1 - l), [spmap[(l-1, m, :inactive)]], [spmap[(l, m, :inactive)]])
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

        switch_active = Catalyst.Reaction(10, [spmap[(l, m, :inactive)]], [spmap[(l, m, :active)]])
        switch_inactive = Catalyst.Reaction(10, [spmap[(l, m, :active)]], [spmap[(l, m, :inactive)]])
        Catalyst.addreaction!(rn, switch_active)
        Catalyst.addreaction!(rn, switch_inactive)

        phosphorylate = Catalyst.Reaction(spmap[(l, m, :active)], [Y], [Yp])
        Catalyst.addreaction!(xn, phosphorylate)
    end

    u0 = zeros(length(spmap) + 3)
    u0[1] = 20
    u0[Catalyst.speciesmap(rn)[spmap[(0, 0, :inactive)]]] = 100

    ps = [20.0, 1.0]
    pr = []
    px = [1.0, 0.1]
    dtimes = 0:0.1:2

    SRXsystem(sn, rn, xn, u0, ps, pr, px, dtimes)
end

system = cooperative_chemotaxis_system()
joint_n = reaction_network(system)

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

act_i = active_indices(joint_n)
inact_i = active_indices(joint_n, "I")
res = _solve(system)

act_sum = vec(sum(res[act_i, :], dims=1))
inact_sum = vec(sum(res[inact_i, :], dims=1))

using Plots
plot(res.t, [act_sum, inact_sum])

g = Catalyst.Graph(system.rn)
g