using Catalyst
using PWS
using StaticArrays
using Test

sn = @reaction_network begin
    κ, ∅ --> L
    λ, L --> ∅
end κ λ

xn = @reaction_network begin
    ρ, L --> L + X
    μ, X --> ∅
end ρ μ

u0 = SA[10, 20]
dtimes = 0:0.5:10.0
ps = [5.0, 1.0]
px = [3.0, 0.1]

system = PWS.SimpleSystem(sn, xn, u0, ps, px, dtimes)

algorithms = [DirectMCEstimate(128), SMCEstimate(128), TIEstimate(0, 4, 128)]
for algorithm in algorithms
    result = mutual_information(system, algorithm, num_responses = 10)
    for v in result[!, :MutualInformation]
        @test v[1] == 0
        @test length(v) == length(system.dtimes)
    end

    initial_condition = PWS.empirical_dist(rand(50, 50), 0:49, 0:49)

    system2 = PWS.SimpleSystem(sn, xn, initial_condition, ps, px, dtimes)
    result2 = mutual_information(system2, algorithm, num_responses = 10)
    for v in result2[!, :MutualInformation]
        @test v[1] != 0
        @test length(v) == length(system.dtimes)
    end
end