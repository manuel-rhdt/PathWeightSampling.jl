# smc_plots.jl
#
# Creates plots for the sequential Monte-Carlo algorithm

import GaussianMcmc
using GaussianMcmc: ConditionalEnsemble, gene_expression_system, marginal_configuration, MarginalEnsemble, SMCEstimate, chemotaxis_system, generate_configuration
using Plots

system = gene_expression_system(dtimes=0:0.1:5.0)
conf = generate_configuration(system)

cond_ens = ConditionalEnsemble(system)
marg_ens = MarginalEnsemble(system)

algorithm = SMCEstimate(256)

particles = []
GaussianMcmc.simulate(algorithm, conf, marg_ens; inspect=x -> push!(particles, x), new_particle=GaussianMcmc.JumpParticleSlow)

function ancestors(particle)
    if particle.parent !== nothing
        list = ancestors(particle.parent)
        push!(list, particle)
        list
    else
        list = [particle]
    end
end

p = Plots.plot()
Plots.plot!(p, conf.s_traj.t, getindex.(conf.s_traj.u, 1), seriestype=:steppre)
# plot!(conf.r_traj.t, getindex.(conf.r_traj.u, 1) ./ 10000)
# plot!(conf.r_traj.t, getindex.(conf.r_traj.u, 2) ./ 10000)
Plots.plot!(conf.x_traj.t, getindex.(conf.x_traj.u, 1))
for ptcl in particles[1]
    # u_vals = hcat(Vector.(getfield.(ancestors(ptcl), :u))...)
    # u_vals ./= [40, 10000, 10000]
    u_vals = hcat(Vector.(getfield.(ancestors(ptcl), :u))...)
    Plots.plot!(p, system.dtimes[2:end], transpose(u_vals), color=:black, legend=false, linewidth=0.1)
end
plot!(p, ylim=(0.40, 0.60))
plot!(p, conf.x_traj)
p

plot()
for ptcl in particles[1]
    plot!(getfield.(ancestors(ptcl), :weight))
end
plot!()

getfield.(ancestors(particles[1][1]), :u)
getfield.(ancestors(particles[1][2]), :u)

system.dtimes


