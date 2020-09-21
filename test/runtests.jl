using Test

@testset "GaussianTests" begin include("gaussian.jl") end
@testset "TrajectoryTests" begin include("trajectories.jl") end
@testset "TrajectoryProbability" begin include("probability.jl") end