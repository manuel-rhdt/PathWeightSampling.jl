using Test

@testset "Events" begin include("events.jl") end
@testset "TrajectoryTests" begin include("trajectories.jl") end
@testset "TrajectoryProbability" begin include("probability.jl") end
@testset "Moves" begin include("moves.jl") end
# @testset "Annealing" begin include("annealing.jl") end
@testset "Gaussian" begin include("gaussian.jl") end
@testset "SRXsystem" begin include("SRXsystem.jl") end
@testset "SMCEstimate" begin include("SMCEstimate.jl") end
