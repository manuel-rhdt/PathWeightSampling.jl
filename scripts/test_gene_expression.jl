import GaussianMcmc
using GaussianMcmc: SMCEstimate, DirectMCEstimate, marginal_configuration, MarginalEnsemble, gene_expression_system, generate_configuration, logpdf

system = gene_expression_system(dtimes=0:0.1:10)
smc = SMCEstimate(50)
dmc = DirectMCEstimate(200)


using Distributed
addprocs(4)

@everywhere begin
    import Pkg
    Pkg.activate(".")
end
@everywhere import GaussianMcmc

result = GaussianMcmc.run_parallel(system, smc, 500)
dresult = GaussianMcmc.run_parallel(system, dmc, 50)

using Plots

plot(system.dtimes, result.MutualInformation, color=:gray, legend=false)
plot!(system.dtimes, dresult.MutualInformation, color=:pink, label="")

using Statistics
plot(system.dtimes, mean(result.MutualInformation), ribbon=sqrt.(var(result.MutualInformation)./size(result, 1)), label="SMC", ylabel="Path mutual information", xlabel="Trajectory length")
plot!(system.dtimes, mean(dresult.MutualInformation), ribbon=sqrt.(var(dresult.MutualInformation)./size(dresult, 1)), label="DMC")

using CSVFiles, DrWatson, DataFrames

zechner_res = DataFrame(load(datadir("zechner/gene_exp.csv")))
plot!(zechner_res.Duration, zechner_res.PMI)


savefig(plot!(dpi=100, size=(6*72,3.5*72)), "~/Downloads/gene_expression.pdf")

size(dresult)



ens = MarginalEnsemble(system)
r = map(1:50) do i
    smc = SMCEstimate(100)
    a = GaussianMcmc.log_marginal(GaussianMcmc.simulate(smc, conf, ens))[end]

    dmc = DirectMCEstimate(100)
    b = GaussianMcmc.log_marginal(GaussianMcmc.simulate(dmc, conf, ens))[end]
    (a, b)
end

r

scatter(randn(50)*0.01, cond .- getindex.(r, 1))
scatter!(randn(50)*0.01 .+ 1, cond .- getindex.(r, 2))

using Statistics
mean(getindex.(r, 1))
mean(getindex.(r, 2))

smc = SMCEstimate(10000)
GaussianMcmc.log_marginal(GaussianMcmc.simulate(smc, conf, ens))[end]

dmc = DirectMCEstimate(10000)
GaussianMcmc.log_marginal(GaussianMcmc.simulate(dmc, conf, ens))[end]


cond = -GaussianMcmc.energy_difference(conf, ens)


histogram(GaussianMcmc.simulate(dmc, conf, ens).samples[end,:])

r = map(1:1000) do i
    result = GaussianMcmc.sample(conf, ens)
    GaussianMcmc.energy_difference(result, ens) => result
end
