import GaussianMcmc
using GaussianMcmc: TrajectoryCallback, SMCEstimate, DirectMCEstimate, marginal_configuration, MarginalEnsemble, ConditionalEnsemble, gene_expression_system, generate_configuration, logpdf
using StaticArrays
import Catalyst
using DiffEqBase
using DiffEqJump
import ModelingToolkit

system_fn = () -> GaussianMcmc.cooperative_chemotaxis_system(delta_e=0, delta_f=-1, lmax=3, mmax=9, γ=1/5)

sol = begin
system = system_fn()
# @time conf = generate_configuration(system)
# units μM
step_s = GaussianMcmc.Trajectory([100.0, 200, 300.0], [SA[15.0], SA[50.0], SA[15.0]], [0])
# conf2 = GaussianMcmc.SRXconfiguration(step_s, conf.r_traj, conf.x_traj)

joint = merge(merge(system.sn, system.rn), system.xn)
rx = merge(system.rn, system.xn)
r_idxs = GaussianMcmc.species_indices(joint, Catalyst.species(system.rn))
u0 = system.u0
u0[1] = step_s.u[1][1]
dprob = DiscreteProblem(rx, u0, (0.0, 100.0), vcat(system.pr, system.px))
jprob = JumpProblem(rx, dprob, Direct(), save_positions=(false, false))

cb = TrajectoryCallback(step_s)
cb = DiscreteCallback(cb, cb, save_positions=(false, false))

sol = solve(jprob, SSAStepper(), callback=cb, tstops=step_s.t, saveat=1.0)
end

using Plots

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

using Transducers
function receptor_states(rs)
    smap = Catalyst.speciesmap(rs)
    xf = KeepSomething() do (species, index) 
        sname = String(ModelingToolkit.operation(species).name)
        mtch = match(r"([A-Z])_(\d+)_(\d+)", sname)
        if mtch !== nothing
            a = mtch.captures[1] == "A"
            l = parse(Int, mtch.captures[2])
            m = parse(Int, mtch.captures[3])
            (a, l, m) => index
        else
            nothing
        end
    end
    smap |> xf |> collect
end

function bound_ligands(sol, rs)
    rstates = receptor_states(rs)
    rstates |> Map(((a, l, m), i)::Pair -> l .* sol[i,:]) |> sum
end

function bound_methyl(sol, rs)
    rstates = receptor_states(rs)
    rstates |> Map(((a, l, m), i)::Pair -> m .* sol[i,:]) |> sum
end

function active_receptors(sol, rs)
    rstates = receptor_states(rs)
    rstates |> Map(((a, l, m), i)::Pair -> a .* sol[i,:]) |> sum
end

begin
t = sol.t
plot(t, vec(sol[1,:]) ./ 100, label="ligand concentration")
plot!(t, active_receptors(sol, joint) ./ 100, label="active fraction")
plot!(t, vec(sol[end-1,:]) ./ 10000, label="Yp / (Y+Yp)")
plot!(t, bound_ligands(sol, joint) ./ 300, label="bound receptor fraction")
plot!(t, bound_methyl(sol, joint) ./ (8*3*100), label="methylated fraction", legend=:topleft)
end


savefig(plot!(dpi=144), "~/Downloads/plot2.png")