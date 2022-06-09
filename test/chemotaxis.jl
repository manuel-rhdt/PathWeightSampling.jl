import PathWeightSampling
using Test

system = PathWeightSampling.cooperative_chemotaxis_system(dtimes=0:0.1:10)
conf = PathWeightSampling.generate_full_configuration(system)
@test length(conf.s_traj[:, 1]) == 1
@test length(conf.r_traj[:, 1]) == 40
@test length(conf.x_traj[:, 1]) == 2

mconf = PathWeightSampling.generate_configuration(system)
@test length(mconf.s_traj[:, 1]) == 1
@test length(mconf.x_traj[:, 1]) == 2

for u in conf.r_traj.u
    @test sum(u) == sum(conf.r_traj.u[1]) # this is true because the total number of receptors must be a constant
end

for u in conf.x_traj.u
    @test sum(u) == sum(conf.x_traj.u[1]) # this is true because the total number of CheY proteins must be a constant
end

cond_ens = PathWeightSampling.ConditionalEnsemble(system)
for t = 0:0.1:10
    local u0 = PathWeightSampling.sample_initial_condition(cond_ens)
    u_new, weight = PathWeightSampling.propagate(mconf, cond_ens, u0, (0.0, t))
    @test size(u0) == size(u_new)
    @test sum(u0[2:end]) == sum(u_new[2:end]) # this is true because the total number of receptors must be a constant
    @test !isinf(weight)
end

algorithm = PathWeightSampling.SMCEstimate(16)
result = PathWeightSampling.mutual_information(system, algorithm, num_samples=1)
@test !all(isinf.(result.MutualInformation[1]))

system_direct_agg = PathWeightSampling.cooperative_chemotaxis_system(dtimes=0:0.1:10, dist_aggregator=PathWeightSampling.GillespieDirect())
@test typeof(system.dist.aggregator) <: PathWeightSampling.DepGraphAggregator
@test typeof(system_direct_agg.dist.aggregator) <: PathWeightSampling.DirectAggregator

cens1 = PathWeightSampling.ConditionalEnsemble(system)
cens2 = PathWeightSampling.ConditionalEnsemble(system_direct_agg)

a = PathWeightSampling.energy_difference(conf, cens1)
b = PathWeightSampling.energy_difference(conf, cens2)
@test a == b

mens1 = PathWeightSampling.MarginalEnsemble(system)
mens2 = PathWeightSampling.MarginalEnsemble(system_direct_agg)

mconf = PathWeightSampling.marginal_configuration(conf)
a = PathWeightSampling.energy_difference(mconf, mens1)
b = PathWeightSampling.energy_difference(mconf, mens2)
@test a == b