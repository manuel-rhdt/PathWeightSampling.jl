using GaussianMcmc
using Distributions
using Test

system = GaussianSystem(delta_t=0.05, duration=1.0)
initial = generate_configuration(system)

n_dim = length(initial) ÷ 2
c_xx = system.joint.Σ[n_dim + 1:end, n_dim + 1:end]

marginal = MvNormal(c_xx)
val1 = logpdf(marginal, initial[n_dim + 1:end])

algorithms = [
    DirectMCEstimate(2^16),
    AnnealingEstimate(10, 50, 2000),
    TIEstimate(2^10, 10, 2^17)
]

for algorithm in algorithms
    result = GaussianMcmc.simulate(algorithm, initial, system, scale=0.6)
    val2 = GaussianMcmc.log_marginal(result)
    @show algorithm (val2 / val1 - 1)
    if :acceptance in fieldnames(typeof(result))
        @show mean(result.acceptance)
    end
    @test isapprox(val1, val2, rtol=1e-2)
end
