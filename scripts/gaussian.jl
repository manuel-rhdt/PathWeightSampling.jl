using LinearAlgebra: iszero
using GaussianMcmc
using Distributions
using LinearAlgebra
using DataFrames
using CSVFiles

data = DataFrame(map(range(0,20,step=0.1)) do duration
    if iszero(duration)
        return (DiscreteTimes=duration, Value=0.0)
    end

    system = GaussianSystem(delta_t=0.05, duration=duration)
    
    n_dim = size(system.joint.Σ, 1) ÷ 2
    c_xx = Hermitian(system.joint.Σ[n_dim + 1:end, n_dim + 1:end])
    c_ss = Hermitian(system.joint.Σ[1:n_dim, 1:n_dim])
    c_z = Hermitian(system.joint.Σ)

    (DiscreteTimes=duration, Value=0.5 * (logdet(c_ss) + logdet(c_xx) - logdet(c_z)))
end)

save(joinpath(pwd(), "plots", "figure_gene_expr", "data", "gene-expression_2021-07-19_gaussian.csv"), data)

marginal = MvNormal(c_xx)
val1 = logpdf(marginal, initial[n_dim + 1:end]) # this is the analytically correct value of the log marginal

