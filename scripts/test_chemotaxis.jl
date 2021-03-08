import GaussianMcmc
using GaussianMcmc: SMCEstimate, DirectMCEstimate, marginal_configuration, MarginalEnsemble, gene_expression_system, generate_configuration, logpdf

system_fn = () -> GaussianMcmc.chemotaxis_system(dtimes=0:0.1:10)
smc = SMCEstimate(16)
dmc = DirectMCEstimate(4)


using Distributed
addprocs(8)

@everywhere begin
    import Pkg
    Pkg.activate(".")
end
@everywhere import GaussianMcmc

system = system_fn()
sol = GaussianMcmc.generate_configuration(system)

@time result = GaussianMcmc.mutual_information(system, smc, num_responses=1)
@time result = GaussianMcmc.run_parallel(system_fn, smc, 80)

using Plots
using Statistics

plot(system.dtimes, result.MutualInformation, color=:gray, legend=false)
plot!(system.dtimes, dresult.MutualInformation, color=:pink, label="")

plot(system.dtimes, mean(result.MutualInformation), ribbon=sqrt.(var(result.MutualInformation)./size(result, 1)), label="SMC", ylabel="Path mutual information", xlabel="Trajectory length")
plot!(system.dtimes, mean(dresult.MutualInformation), ribbon=sqrt.(var(dresult.MutualInformation)./size(dresult, 1)), label="DMC")
