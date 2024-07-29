using Test
import PathWeightSampling as PWS

# doctest(PathWeightSampling)

@testset "PathWeightSampling.jl" begin
    @testset "TrajectoryTests" begin
        include("trajectories.jl")
    end
    @testset "TrajectoryProbability" begin
        include("probability.jl")
    end
    @testset "SSA" begin
        include("ssa.jl")
    end
    @testset "SimpleSystem" begin
        include("SimpleSystem.jl")
    end
    @testset "HybridSystem" begin
        include("HybridSystem.jl")
    end
    @testset "Three Species" begin
        include("three_species.jl")
    end
    @testset "SMCEstimate" begin
        include("SMCEstimate.jl")
    end
    @testset "Chemotaxis" begin
        include("chemotaxis.jl")
    end
end