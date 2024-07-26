import PathWeightSampling as PWS
using Test
using Statistics

import Random: Xoshiro

system = PWS.chemotaxis_system(n=3, n_clusters=800, duration=1.0, dt=0.1)

conf = PWS.generate_configuration(system, rng=Xoshiro(1))

@test size(conf.traj)[1] == PWS.SSA.num_species(system.reactions)

algs = [PWS.SMCEstimate(256), PWS.DirectMCEstimate(256), PWS.PERM(32)]

result = Dict(map(algs) do alg
    num_samples = 24
    mi = Vector{Vector{Float64}}(undef, num_samples)
    Threads.@threads for i in eachindex(mi)
        @time "Generate Sample $i/$num_samples with $(PWS.name(alg))" begin
            cresult = PWS.conditional_density(system, alg, conf)
            mresult = PWS.marginal_density(system, alg, conf)
            mi[i] = cresult - mresult
        end
    end
    PWS.name(alg) => reduce(hcat, mi)
end)

# Test that all results lie within one standard deviations of each other.
# This should find coarse implementation errors for specific estimates.
for (name, mi) in result
    lower_bound = mean(mi, dims=2) - std(mi, dims=2)
    for (name2, mi2) in result
        upper_bound = mean(mi2, dims=2) + std(mi2, dims=2)
        @test all(lower_bound .<= upper_bound)
    end
end


# # Some plotting routines are commented out below to see what goes wrong.
# # Use them to see what's wrong when the tests above fail.
#--
# begin
#     import Plots
#     Plots.plot()
#     colormap = Dict("SMC" => 1, "Direct MC" => 2, "PERM" => 3)
#     dtimes = PWS.discrete_times(system)
#     for (name, mi) in result
#         Plots.plot!(dtimes, mean(mi, dims=2), ribbon=std(mi, dims=2), color=colormap[name], label=name)
#         Plots.plot!(dtimes, mi, color=colormap[name], linewidth=0.3, label="")
#     end
#     Plots.plot!()
# end
# --