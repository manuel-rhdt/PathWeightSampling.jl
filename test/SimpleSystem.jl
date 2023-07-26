import PathWeightSampling as PWS
using Test

system = PWS.simple_chemotaxis_system(n=3, n_clusters=800, duration=2.0, dt=0.1)
dtimes = PWS.discrete_times(system)

algorithms = [PWS.DirectMCEstimate(128), PWS.SMCEstimate(128), PWS.PERM(128)]
for algorithm in algorithms
    result = PWS.mutual_information(system, algorithm, num_samples=4)
    for v in result[!, :MutualInformation]
        @test v[1] == 0
        @test length(v) == length(dtimes)
    end
end


# system = PWS.simple_chemotaxis_system(n=15, n_clusters=20, duration=20.0, dt=0.1)

# alg = PWS.PERM(10)

# conf = PWS.generate_configuration(system)
# @time PWS.marginal_density(system, alg, conf)
# @time PWS.conditional_density(system, alg, conf)
