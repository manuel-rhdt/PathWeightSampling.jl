import PWS
using Test

system = PWS.cooperative_chemotaxis_system(dtimes=0:0.1:10)
conf = PWS.generate_configuration(system)
@test length(conf.s_traj.u[1]) == 1
@test length(conf.r_traj.u[1]) == 40
@test length(conf.x_traj.u[1]) == 2

for u in conf.r_traj.u
    @test sum(u) == sum(conf.r_traj.u[1]) # this is true because the total number of receptors must be a constant
end

for u in conf.x_traj.u
    @test sum(u) == sum(conf.x_traj.u[1]) # this is true because the total number of CheY proteins must be a constant
end

cond_ens = PWS.ConditionalEnsemble(system)
for t in 0:0.1:10
    u0 = PWS.sample_initial_condition(cond_ens)
    u_new, weight = PWS.propagate(conf, cond_ens, u0, (0.0, t))
    @test size(u0) == size(u_new)
    @test sum(u0[2:end]) == sum(u_new[2:end]) # this is true because the total number of receptors must be a constant
    @test !isinf(weight)
end

algorithm = PWS.SMCEstimate(128)
result = PWS.mutual_information(system, algorithm, num_samples=1)
@test !all(isinf.(result.MutualInformation[1]))
