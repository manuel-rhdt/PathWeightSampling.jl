using Distributed
@everywhere import PWS

system_fn = () -> PWS.gene_expression_system(dtimes=0:0.1:10)
smc = PWS.SMCEstimate(128)
result = PWS.run_parallel(system_fn, smc, 32)
println(result)
