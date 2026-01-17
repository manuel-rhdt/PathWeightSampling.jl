"""
    constants.jl

Centralized test configuration and constants.
Reduces magic numbers scattered throughout test files and makes it easy to
adjust test parameters globally.
"""

# ============================================================================
# Particle Counts for Estimation Algorithms
# ============================================================================

"""
    NUM_PARTICLES_DEFAULT

Default number of particles for SMC and Direct MC estimation algorithms.
Used in most algorithm comparison tests.
"""
const NUM_PARTICLES_DEFAULT = 256

"""
    NUM_PARTICLES_PERM

Number of particles for PERM algorithm (typically lower than other methods).
"""
const NUM_PARTICLES_PERM = 32

"""
    NUM_PARTICLES_SMALL

Minimal number of particles for quick smoke tests.
"""
const NUM_PARTICLES_SMALL = 16

# ============================================================================
# Sample Counts
# ============================================================================

"""
    NUM_SAMPLES_STANDARD

Standard number of samples for statistical estimation tests.
"""
const NUM_SAMPLES_STANDARD = 24

"""
    NUM_SAMPLES_QUICK

Reduced number of samples for quick validation tests.
"""
const NUM_SAMPLES_QUICK = 4

"""
    NUM_SAMPLES_EXTENDED

Extended number of samples for comprehensive statistical tests.
Used when testing convergence or stability properties.
"""
const NUM_SAMPLES_EXTENDED = 500

# ============================================================================
# System Configuration Parameters
# ============================================================================

"""
    SYSTEM_DURATION_SHORT

Short duration for quick system tests (seconds).
"""
const SYSTEM_DURATION_SHORT = 1.0

"""
    SYSTEM_DURATION_MEDIUM

Medium duration for comprehensive system tests (seconds).
"""
const SYSTEM_DURATION_MEDIUM = 10.0

"""
    SYSTEM_DURATION_LONG

Long duration for extended system tests (seconds).
"""
const SYSTEM_DURATION_LONG = 100.0

"""
    SYSTEM_DT

Standard time step for discrete observation points (seconds).
"""
const SYSTEM_DT = 0.1

"""
    CHEMOTAXIS_N_CLUSTERS

Number of clusters in chemotaxis system test fixture.
"""
const CHEMOTAXIS_N_CLUSTERS = 800

"""
    CHEMOTAXIS_N_STANDARD

Standard dimension parameter for chemotaxis system.
"""
const CHEMOTAXIS_N_STANDARD = 3

"""
    CHEMOTAXIS_N_EXTENDED

Extended dimension parameter for chemotaxis system comprehensive tests.
"""
const CHEMOTAXIS_N_EXTENDED = 6

# ============================================================================
# Random Number Generation
# ============================================================================

"""
    RANDOM_SEED_BASE

Base random seed for reproducible tests.
Each test typically uses RANDOM_SEED_BASE + test_index to avoid collision.
"""
const RANDOM_SEED_BASE = 1

# ============================================================================
# Statistical Testing Parameters
# ============================================================================

"""
    RELATIVE_TOLERANCE_STRICT

Strict relative tolerance for comparing exact computations.
"""
const RELATIVE_TOLERANCE_STRICT = 1e-10

"""
    RELATIVE_TOLERANCE_NORMAL

Normal relative tolerance for algorithm comparisons.
"""
const RELATIVE_TOLERANCE_NORMAL = 0.1

"""
    RELATIVE_TOLERANCE_RELAXED

Relaxed relative tolerance for statistical estimates with natural variance.
"""
const RELATIVE_TOLERANCE_RELAXED = 0.5

"""
    ABSOLUTE_TOLERANCE_SMALL

Small absolute tolerance for floating point comparisons.
"""
const ABSOLUTE_TOLERANCE_SMALL = 1e-10

"""
    ABSOLUTE_TOLERANCE_MEDIUM

Medium absolute tolerance for statistical estimates.
"""
const ABSOLUTE_TOLERANCE_MEDIUM = 0.1

"""
    ABSOLUTE_TOLERANCE_LARGE

Large absolute tolerance for tests comparing values with inherent variance.
"""
const ABSOLUTE_TOLERANCE_LARGE = 0.5

# ============================================================================
# Algorithm-Specific Parameters
# ============================================================================

"""
    CHEMOTAXIS_PHOSPHORYLATION_RTOL

Relative tolerance for chemotaxis phosphorylation rate estimation.
"""
const CHEMOTAXIS_PHOSPHORYLATION_RTOL = 0.05

"""
    THREE_SPECIES_MI_EXPECTED

Expected mutual information value for three-species cascade at final time.
"""
const THREE_SPECIES_MI_EXPECTED = 1.5

"""
    THREE_SPECIES_MI_TOLERANCE

Absolute tolerance for three-species mutual information estimate.
"""
const THREE_SPECIES_MI_TOLERANCE = 0.5
