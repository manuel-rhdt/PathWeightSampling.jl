"""
    fixtures.jl

Reusable test fixtures for common system configurations and trajectories.
These fixtures reduce duplication across test files and ensure consistency
in test setup across the test suite.
"""

import PathWeightSampling as PWS
import Random
using StaticArrays

# ============================================================================
# Chemotaxis System Fixtures
# ============================================================================

"""
    get_chemotaxis_system(; n=3, n_clusters=800, duration=1.0, dt=0.1)

Create a standard chemotaxis system fixture for testing.
Commonly used in SMCEstimate, Chemotaxis, and SimpleSystem tests.
"""
function get_chemotaxis_system(; n=3, n_clusters=800, duration=1.0, dt=0.1)
    PWS.chemotaxis_system(n=n, n_clusters=n_clusters, duration=duration, dt=dt)
end

"""
    get_chemotaxis_system_extended(; n=6, n_clusters=800, duration=100.0, dt=0.1)

Create an extended chemotaxis system for comprehensive testing.
Used in Chemotaxis test with extended parameter sweep.
"""
function get_chemotaxis_system_extended(; n=6, n_clusters=800, duration=100.0, dt=0.1)
    PWS.chemotaxis_system(n=n, n_clusters=n_clusters, duration=duration, dt=dt)
end

# ============================================================================
# Birth-Death Process Fixtures
# ============================================================================

"""
    get_birth_death_system(; u0=50.0, duration=100.0)

Create a simple birth-death process system for testing trajectory generation.
"""
function get_birth_death_system(; u0=50.0, duration=100.0)
    rates = [50.0, 1.0]
    rstoich = [Pair{Int, Int}[], [1 => 1]]
    nstoich = [[1 => 1], [1 => -1]]

    bd_reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:X, :Y])

    PWS.MarkovJumpSystem(
        PWS.GillespieDirect(),
        bd_reactions,
        u0,
        (0.0, duration),
        :Y,
        :X
    )
end

"""
    get_birth_death_system_for_probability()

Create a birth-death system with specific parameters for probability computation tests.
Static array initialization and precise time points.
"""
function get_birth_death_system_for_probability()
    κ = 50.0
    λ = 1.0

    species = [:X, :Y]
    rates = SA[κ, λ]
    rstoich = [Pair{Int, Int}[], [1 => 1]]
    nstoich = [[1 => 1], [1 => -1]]

    bd_reactions = PWS.ReactionSet(rates, rstoich, nstoich, species)

    u0 = SA[50.0, 0.0]
    tspan = (0.0, 3.0)

    PWS.MarkovJumpSystem(
        PWS.GillespieDirect(),
        bd_reactions,
        u0,
        tspan,
        :Y,
        :X,
        0.1
    )
end

# ============================================================================
# Three-Species Cascade Fixtures
# ============================================================================

# Define the rate functor type at module level
struct ThreeSpeciesRates
    κ::Float32
    λ::Float32
    ρ::Float32
    μ::Float32
    ρ2::Float32
    μ2::Float32
end

function (rates::ThreeSpeciesRates)(rxidx, u::AbstractVector)
    if rxidx == 1
        rates.κ
    elseif rxidx == 2
        u[1] * rates.λ
    elseif rxidx == 3
        u[1] * rates.ρ
    elseif rxidx == 4
        u[2] * rates.μ
    elseif rxidx == 5
        u[2] * rates.ρ2
    elseif rxidx == 6
        u[3] * rates.μ2
    else
        0.0
    end
end

"""
    get_three_species_system()

Create a three-species cascade system (S → V → X) for testing cascade behavior.
"""
function get_three_species_system()
    κ = 10.0
    λ = 1.0
    ρ = 1.0
    μ = 1.0
    ρ2 = 10.0
    μ2 = 10.0

    rstoich = [
        Pair{Int, Int}[],
        [1 => 1],
        [1 => 1],
        [2 => 1],
        [2 => 1],
        [3 => 1]
    ]
    nstoich = [
        [1 => 1],
        [1 => -1],
        [2 => 1],
        [2 => -1],
        [3 => 1],
        [3 => -1]
    ]
    species = [:S, :V, :X]

    rates = ThreeSpeciesRates(κ, λ, ρ, μ, ρ2, μ2)

    jumps = PWS.SSA.ConstantRateJumps(rates, rstoich, nstoich, species)
    u0 = SA{Int16}[
        κ / λ,
        κ / λ * ρ / μ,
        κ / λ * ρ / μ * ρ2 / μ2
    ]
    tspan = (0.0, 10.0)

    PWS.MarkovJumpSystem(
        PWS.GillespieDirect(),
        jumps,
        u0,
        tspan,
        :S,
        :X,
        1e-1
    )
end

# ============================================================================
# Sample Traces for Testing
# ============================================================================

"""
    get_sample_reaction_trace()

Create a minimal reaction trace for testing probability computations.
"""
function get_sample_reaction_trace()
    PWS.ReactionTrace([1.0, 2.0], [1, 2], BitSet([1, 2]))
end

"""
    get_sample_hybrid_trace()

Create a minimal hybrid trace for testing hybrid system probability computations.
"""
function get_sample_hybrid_trace()
    dtimes = [1.0, 2.0]
    u = [[51.0, 0.0], [50.0, 0.0]]
    PWS.SSA.HybridTrace(Float64[], Int16[], BitSet([1, 2]), u, dtimes, [1 => 1])
end

# ============================================================================
# Helper Functions for Configuration Generation
# ============================================================================

"""
    get_configuration(system; rng_seed=1)

Generate a configuration (trace and trajectory) for a system with reproducible RNG.
"""
function get_configuration(system; rng_seed=1)
    PWS.generate_configuration(system, rng=Random.Xoshiro(rng_seed))
end

"""
    get_configurations_batch(system, num_configs; start_seed=1)

Generate multiple configurations with different random seeds.
"""
function get_configurations_batch(system, num_configs; start_seed=1)
    [get_configuration(system; rng_seed=start_seed + i - 1) for i in 1:num_configs]
end
