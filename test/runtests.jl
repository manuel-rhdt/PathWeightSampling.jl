using Distributions: Gaussian
using Test, Documenter, PathWeightSampling

doctest(PathWeightSampling)
@testset "Events" begin include("events.jl") end
@testset "TrajectoryTests" begin include("trajectories.jl") end
@testset "TrajectoryProbability" begin include("probability.jl") end
@testset "Configurations" begin include("configurations.jl") end
@testset "SimpleSystem" begin include("SimpleSystem.jl") end
@testset "ComplexSystem" begin include("ComplexSystem.jl") end
@testset "SMCEstimate" begin include("SMCEstimate.jl") end
@testset "MCMCMoves" begin include("mcmc_moves.jl") end
@testset "Gaussian" begin include("gaussian.jl") end
# @testset "Chemotaxis" begin include("chemotaxis.jl") end
