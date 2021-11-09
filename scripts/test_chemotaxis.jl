import PWS
using PWS: SMCEstimate, DirectMCEstimate, marginal_configuration, ConditionalEnsemble, MarginalEnsemble, gene_expression_system, generate_configuration, log_marginal, logpdf, simulate

smc = SMCEstimate(16)
dmc = DirectMCEstimate(4)


# using Distributed
# addprocs(8)

# @everywhere begin
#     import Pkg
#     Pkg.activate(".")
# end
# @everywhere import PWS

system = PWS.chemotaxis_system(mean_L=50, dtimes=0:0.1:10.0)
@time conf = generate_configuration(system)

cens = ConditionalEnsemble(system)
mens = MarginalEnsemble(system)
mconf = marginal_configuration(conf)

cr = simulate(smc, conf, cens)
mr = simulate(smc, mconf, mens)
log_marginal(cr) - log_marginal(mr)


result = PWS.mutual_information(system, smc, num_samples=10)


# @time result = PWS.run_parallel(system_fn, smc, 80)

using Plots
# using Statistics
plot(system.dtimes, result.MutualInformation, color=:gray, legend=false)
# plot!(system.dtimes, dresult.MutualInformation, color=:pink, label="")

# plot(system.dtimes, mean(result.MutualInformation), ribbon=sqrt.(var(result.MutualInformation)./size(result, 1)), label="SMC", ylabel="Path mutual information", xlabel="Trajectory length")
# plot!(system.dtimes, mean(dresult.MutualInformation), ribbon=sqrt.(var(dresult.MutualInformation)./size(dresult, 1)), label="DMC")
