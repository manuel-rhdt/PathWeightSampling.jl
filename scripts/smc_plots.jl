import GaussianMcmc
using GaussianMcmc: ConditionalEnsemble, marginal_configuration, MarginalEnsemble, SMCEstimate, chemotaxis_system, generate_configuration
using Plots

system = chemotaxis_system(dtimes=0:0.01:2)
conf = generate_configuration(system)

plot(conf.r_traj)

cond_ens = ConditionalEnsemble(system)
marg_ens = MarginalEnsemble(system)

algorithm = SMCEstimate(200)

particles = []
GaussianMcmc.simulate(algorithm, marginal_configuration(conf), marg_ens; inspect=x -> push!(particles, x))

function ancestors(particle)
    if particle.parent !== nothing
        list = ancestors(particle.parent)
        push!(list, particle)
        list
    else
        list = [particle]
    end
end

p = plot(conf.r_traj)
plot!(p, conf.s_traj)
for ptcl in particles[1]
    u_vals = hcat(Vector.(getfield.(ancestors(ptcl), :u))...)
    plot!(p, system.dtimes[2:end], transpose(u_vals), legend=false, color=:black, linewidth=0.1)
end
p
plot!(p, conf.x_traj)
p

getfield.(ancestors(particles[1][1]), :u)
getfield.(ancestors(particles[1][2]), :u)

system.dtimes


