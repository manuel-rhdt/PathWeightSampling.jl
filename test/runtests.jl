using Distributions: Gaussian
using Test, Documenter
import PathWeightSampling as PWS

# doctest(PathWeightSampling)
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
@testset "SMCEstimate" begin
    include("SMCEstimate.jl")
end
@testset "Gaussian" begin
    include("gaussian.jl")
end
@testset "Chemotaxis" begin
    include("chemotaxis.jl")
end
