import PWS
using PWS: SMCEstimate, DirectMCEstimate, TIEstimate, marginal_configuration, MarginalEnsemble, gene_expression_system, generate_configuration, logpdf

system_fn = () -> PWS.gene_expression_system(dtimes=0:0.1:10)
smc = SMCEstimate(256)
dmc = DirectMCEstimate(8*256)
ti  = TIEstimate(0, 8, 256)

system = system_fn()

conf = generate_configuration(system)

sresult = PWS.mutual_information(system, smc, num_samples=200)
tresult = PWS.mutual_information(system, ti, num_samples=5)
dresult= PWS.mutual_information(system, dmc, num_samples=5)

using Plots
plot(conf.s_traj, xlim=(0,2), seriescolor=:green, label="S")
plot!(conf.x_traj, seriescolor=:cornflowerblue, label="X")
savefig("~/Downloads/example_traj.pdf")

plot(system.dtimes, result.MutualInformation, color=:gray, legend=false)
plot!(system.dtimes, dresult.MutualInformation, color=:pink, label="")

using Statistics
plot(system.dtimes, mean(result.MutualInformation), ribbon=sqrt.(var(result.MutualInformation)./size(result, 1)), label="SMC", ylabel="Path mutual information", xlabel="Trajectory length")
plot!(system.dtimes, mean(dresult.MutualInformation), ribbon=sqrt.(var(dresult.MutualInformation)./size(dresult, 1)), label="DMC")
plot!(system.dtimes, mean(tresult.MutualInformation), ribbon=sqrt.(var(tresult.MutualInformation)./size(tresult, 1)), label="TI")


using CSVFiles, DrWatson, DataFrames

zechner_res = DataFrame(load(datadir("zechner/gene_exp.csv")))
plot!(zechner_res.Duration[1:20], zechner_res.PMI[1:20])


savefig(plot!(dpi=100, size=(6*72,3.5*72)), "~/Downloads/gene_expression.pdf")


ens = MarginalEnsemble(system)
r = map(1:50) do i
    smc = SMCEstimate(100)
    a = PWS.log_marginal(PWS.simulate(smc, conf, ens))[end]

    dmc = DirectMCEstimate(100)
    b = PWS.log_marginal(PWS.simulate(dmc, conf, ens))[end]
    (a, b)
end

r

scatter(randn(50)*0.01, cond .- getindex.(r, 1))
scatter!(randn(50)*0.01 .+ 1, cond .- getindex.(r, 2))

using Statistics
mean(getindex.(r, 1))
mean(getindex.(r, 2))

smc = SMCEstimate(10000)
PWS.log_marginal(PWS.simulate(smc, conf, ens))[end]

dmc = DirectMCEstimate(10000)
PWS.log_marginal(PWS.simulate(dmc, conf, ens))[end]


cond = -PWS.energy_difference(conf, ens)


histogram(PWS.simulate(dmc, conf, ens).samples[end,:])

r = map(1:1000) do i
    result = PWS.sample(conf, ens)
    PWS.energy_difference(result, ens) => result
end
