"""
    runtests.jl

Main test orchestrator for PathWeightSampling.jl
Organizes tests into three categories: unit tests, integration tests, and
algorithm comparison tests. Also validates documentation examples via doctest.
"""

using Test
import PathWeightSampling as PWS

using Documenter

# Include shared test utilities
include("constants.jl")
include("fixtures.jl")
include("helpers.jl")

# Run doctests from package documentation
doctest(PWS)

@testset "PathWeightSampling.jl" begin

    # ========================================================================
    # Unit Tests: Core algorithmic components
    # ========================================================================
    @testset "Unit Tests" begin
        @testset "Trajectory Generation" begin
            include("unit/trajectory_generation.jl")
        end
        @testset "Probability Computation" begin
            include("unit/probability_computation.jl")
        end
        @testset "Stochastic Simulation Algorithm" begin
            include("unit/stochastic_simulation_algorithm.jl")
        end
    end

    # ========================================================================
    # Integration Tests: Multi-component systems
    # ========================================================================
    @testset "Integration Tests" begin
        @testset "Simple System" begin
            include("integration/simple_system.jl")
        end
        @testset "Hybrid System (SDE + Jumps)" begin
            include("integration/hybrid_system.jl")
        end
        @testset "Three Species Cascade" begin
            include("integration/three_species_cascade.jl")
        end
        @testset "Chemotaxis System" begin
            include("integration/chemotaxis_system.jl")
        end
    end

    # ========================================================================
    # Algorithm Tests: Estimation and sampling algorithms
    # ========================================================================
    @testset "Algorithm Tests" begin
        @testset "SMC Estimation" begin
            include("algorithms/smc_estimate.jl")
        end
    end

end