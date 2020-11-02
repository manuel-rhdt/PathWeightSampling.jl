using Test

@testset "GaussianTests" begin include("gaussian.jl") end
@testset "TrajectoryTests" begin include("trajectories.jl") end
@testset "TrajectoryProbability" begin include("probability.jl") end
@testset "Moves" begin include("moves.jl") end
@testset "Annealing" begin include("annealing.jl") end
@testset "Stationary" begin include("stationary.jl") end