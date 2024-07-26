import PathWeightSampling as PWS
using Test

system = PWS.chemotaxis_system(n=3, n_clusters=800, duration=2.0, dt=0.1)
dtimes = PWS.discrete_times(system)

algorithms = [PWS.DirectMCEstimate(128), PWS.SMCEstimate(128), PWS.PERM(128)]
for algorithm in algorithms
    mi = PWS.mutual_information(system, algorithm, num_samples=4)
    @test length(mi.result.MutualInformation) == 4 * length(dtimes)
    @test mi.result.MutualInformation[mi.result.time .== 0] == [0, 0, 0, 0]
end
