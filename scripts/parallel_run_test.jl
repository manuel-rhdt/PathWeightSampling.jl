using Distributed
@everywhere import GaussianMcmc

system_fn = () -> GaussianMcmc.gene_expression_system(dtimes=0:0.1:10)
smc = GaussianMcmc.SMCEstimate(128)
result = GaussianMcmc.run_parallel(system_fn, smc, 32)
println(result)
