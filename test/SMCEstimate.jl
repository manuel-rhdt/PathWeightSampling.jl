import GaussianMcmc
using GaussianMcmc: SMCEstimate, name, propagate, DirectMCEstimate, marginal_configuration, ConditionalEnsemble, MarginalEnsemble, energy_difference, generate_configuration, log_marginal, logpdf, simulate
using Test
using Statistics

system = GaussianMcmc.chemotaxis_system(dtimes=0:0.025:0.5)

@test system.dist.aggregator.update_map == [0, 0, 0, 0, 1, 2]

conf = generate_configuration(system)
mconf = marginal_configuration(conf)
cens = ConditionalEnsemble(system)
mens = MarginalEnsemble(system)

@test energy_difference(conf, cens) == energy_difference(mconf, mens)

algs = [SMCEstimate(16), DirectMCEstimate(16)]

result = Dict(map(algs) do alg
    mi = map(1:20) do i
        cresult = simulate(alg, conf, cens)
        mresult = simulate(alg, mconf, mens)
        log_marginal(cresult) - log_marginal(mresult)
    end
    name(alg) => hcat(mi...)
end)

# test that all results lie within 3 standard deviations from each other
# this should find most implementation errors for specific estimates
for (name, mi) in result
    lower_bound = mean(mi, dims=2) - 3*std(mi, dims=2)
    for (name2, mi2) in result
        upper_bound = mean(mi2, dims=2) + 3*std(mi2, dims=2)
        @test all(lower_bound .<= upper_bound)
    end
end

# # Some plotting routines are commented out below to see what goes wrong.
# # Use them to see what's wrong when the tests above fail.

# plot()
# colormap = Dict("SMC" => 1, "Direct MC" => 2)
# for (name, mi) in result
#     plot!(system.dtimes, mean(mi, dims=2), ribbon=3*std(mi, dims=2), color=colormap[name], label=name)
#     plot!(system.dtimes, mi, color=colormap[name], linewidth=0.3, label="")
# end
# plot!()