import PathWeightSampling
using PathWeightSampling: SMCEstimate, name, propagate, DirectMCEstimate, marginal_configuration, ConditionalEnsemble, MarginalEnsemble, energy_difference, trajectory_energy, generate_configuration, log_marginal, logpdf, simulate
using Test
using Statistics

system = PathWeightSampling.chemotaxis_system(dtimes = 0:0.05:2.0)

@test system.dist.aggregator.update_map == [0, 0, 0, 0, 1, 2]

conf = PathWeightSampling.generate_full_configuration(system)
mconf = marginal_configuration(conf)
cens = ConditionalEnsemble(system)
mens = MarginalEnsemble(system)

@test energy_difference(conf, cens) == energy_difference(mconf, mens)

merged = PathWeightSampling.merge_trajectories(conf.s_traj, conf.r_traj, conf.x_traj)
for (i, T) in enumerate(system.dtimes)
    @test energy_difference(conf, cens)[i] ≈ -trajectory_energy(cens.dist, merged, tspan = (0.0, T))
    @test energy_difference(conf, cens)[i] ≈ -trajectory_energy(mens.dist, merged, tspan = (0.0, T))

    if i >= 2
        E(i) = energy_difference(conf, cens)[i]
        T(i) = system.dtimes[i]
        dE = E(i - 1) - E(i)
        @test dE ≈ trajectory_energy(mens.dist, merged, tspan = (T(i - 1), T(i)))
        @test dE ≈ trajectory_energy(cens.dist, merged, tspan = (T(i - 1), T(i)))
    end
end


algs = [SMCEstimate(256), DirectMCEstimate(256)]

result = Dict(map(algs) do alg
    mi = map(1:20) do i
        @time begin
            cresult = simulate(alg, conf, cens)
            mresult = simulate(alg, mconf, mens)
            log_marginal(cresult) - log_marginal(mresult)
        end
    end
    name(alg) => hcat(mi...)
end)

# Test that all results lie within 2 standard deviations of each other.
# This should find coarse implementation errors for specific estimates.
for (name, mi) in result
    lower_bound = mean(mi, dims = 2) - 2 * std(mi, dims = 2)
    for (name2, mi2) in result
        upper_bound = mean(mi2, dims = 2) + 2 * std(mi2, dims = 2)
        @test all(lower_bound .<= upper_bound)
    end
end


# # Some plotting routines are commented out below to see what goes wrong.
# # Use them to see what's wrong when the tests above fail.
#--
# using Plots
# plot()
# colormap = Dict("SMC" => 1, "Direct MC" => 2)
# for (name, mi) in result
#     plot!(system.dtimes, mean(mi, dims=2), ribbon=3*std(mi, dims=2), color=colormap[name], label=name)
#     plot!(system.dtimes, mi, color=colormap[name], linewidth=0.3, label="")
# end
# plot!()
# --