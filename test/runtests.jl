using Distributions: Gaussian
using Test, Documenter, PWS

doctest(PWS)
@testset "Events" begin include("events.jl") end
@testset "TrajectoryTests" begin include("trajectories.jl") end
@testset "TrajectoryProbability" begin include("probability.jl") end
@testset "SimpleSystem" begin include("SimpleSystem.jl") end
@testset "ComplexSystem" begin include("ComplexSystem.jl") end
@testset "SMCEstimate" begin include("SMCEstimate.jl") end
@testset "MCMCMoves" begin include("mcmc_moves.jl") end
@testset "Gaussian" begin include("gaussian.jl") end
