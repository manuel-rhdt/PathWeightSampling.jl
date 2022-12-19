import PathWeightSampling as PWS
import PathWeightSampling: GaussianSystem
using Distributions
using Test
using LinearAlgebra

system = GaussianSystem(delta_t=0.05, duration=1.0)
initial = PWS.generate_configuration(system)

n_dim = length(initial) ÷ 2
c_xx = system.joint.Σ[n_dim + 1:end, n_dim + 1:end]
@test issymmetric(c_xx)

marginal = MvNormal(c_xx)
val1 = logpdf(marginal, initial[n_dim + 1:end]) # this is the analytically correct value of the log marginal

# we now test for every implemented estimation algorithm whether they yield results within 1% of the correct value

algorithms = [
    PWS.DirectMCEstimate(2^16),
    PWS.AnnealingEstimate(10, 50, 2000),
    PWS.TIEstimate(2^10, 10, 2^17)
]

for algorithm in algorithms
    local result = PWS.simulate(algorithm, initial, system, scale=0.6)
    val2 = PWS.log_marginal(result)
    @show algorithm (val2 / val1 - 1)
    if :acceptance in fieldnames(typeof(result))
        @show mean(result.acceptance)
    end
    @test isapprox(val1, val2, rtol=1e-2)
end
