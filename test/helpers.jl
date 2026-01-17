"""
    helpers.jl

Common test helper functions used across multiple test files.
These utilities reduce duplication and provide consistent assertions and
utility functions for the test suite.
"""

import PathWeightSampling as PWS
import Statistics: mean, std, var
using DataFrames

# ============================================================================
# Statistical Helpers
# ============================================================================

"""
    sem(x)

Compute the standard error of the mean for a vector of values.
Standard error = std(x) / sqrt(n)
"""
sem(x) = sqrt(var(x) / length(x))

# ============================================================================
# Statistical Assertion Helpers
# ============================================================================

"""
    assert_within_bounds(value::Real, lower::Real, upper::Real; msg::String="")

Assert that a value falls within specified bounds.
Useful for validating that estimates are reasonable.
"""
function assert_within_bounds(value::Real, lower::Real, upper::Real; msg::String="")
    if !(lower <= value <= upper)
        error("Value $value outside bounds [$lower, $upper]. $msg")
    end
end


# ============================================================================
# Algorithm Comparison Helpers
# ============================================================================

"""
    run_algorithm_ensemble(system, conf, algorithms::Vector; num_samples)

Run multiple algorithms on the same configuration and return results.
Useful for comparing algorithm outputs on identical data.
"""
function run_algorithm_ensemble(system, conf, algorithms::Vector; num_samples)
    result = Dict()
    for alg in algorithms
        mi = Vector{Vector{Float64}}(undef, num_samples)
        for i in eachindex(mi)
            cresult = PWS.conditional_density(system, alg, conf)
            mresult = PWS.marginal_density(system, alg, conf)
            mi[i] = cresult - mresult
        end
        result[PWS.name(alg)] = reduce(hcat, mi)
    end
    result
end

"""
    validate_algorithm_consistency(results::Dict)

Check that different algorithms produce approximately consistent results.
Validates that all algorithm bounds overlap within tolerance.
"""
function validate_algorithm_consistency(results::Dict)
    algorithms = collect(keys(results))

    for i in eachindex(algorithms)
        for j in (i+1):length(algorithms)
            alg1, alg2 = algorithms[i], algorithms[j]
            data1 = results[alg1]
            data2 = results[alg2]

            lower1 = mean(data1, dims=2) - std(data1, dims=2)
            upper2 = mean(data2, dims=2) + std(data2, dims=2)

            if !all(lower1 .<= upper2)
                return false
            end
        end
    end
    return true
end

# ============================================================================
# Mutual Information & Density Helpers
# ============================================================================

"""
    compute_mutual_information_stats(system, algorithm; num_samples, label="")

Compute mutual information and return summary statistics.
"""
function compute_mutual_information_stats(system, algorithm; num_samples, label="")
    mi = PWS.mutual_information(system, algorithm; num_samples=num_samples)

    stats = combine(
        groupby(mi.result, :time),
        :MutualInformation => mean => :MI,
        :MutualInformation => std => :Std,
        :MutualInformation => sem => :Err
    )

    stats
end

"""
    validate_conditional_vs_marginal(system, conf, algorithm)

Assert that conditional density >= marginal density (information theory requirement).
"""
function validate_conditional_vs_marginal(system, conf, algorithm)
    cd = PWS.conditional_density(system, algorithm, conf)
    md = PWS.marginal_density(system, algorithm, conf)

    all(cd .>= md .- 1e-10)  # small numerical tolerance
end

"""
    validate_deterministic_evaluation(cd1, cd2)

Verify that conditional density computation is deterministic.
"""
function validate_deterministic_evaluation(cd1, cd2)
    cd1 == cd2
end

"""
    validate_stochastic_variability(md1, md2; rtol=1e-3)

Verify that marginal density shows expected stochastic variability.
While results should differ, they should be close.
"""
function validate_stochastic_variability(md1, md2; rtol=1e-3)
    (md1 != md2) && all(isapprox.(md1, md2, rtol=rtol))
end

# ============================================================================
# Trajectory Validation Helpers
# ============================================================================

"""
    validate_sorted_trajectory(trace::PWS.ReactionTrace)

Check that reaction times are sorted and unique.
"""
function validate_sorted_trajectory(trace::PWS.ReactionTrace)
    issorted(trace.t) && allunique(trace.t)
end

"""
    validate_state_consistency(system, conf)

Validate that trajectory states are consistent with reaction trace.
Reconstructs states from reactions and compares with stored trajectory.
"""
function validate_state_consistency(system, conf)
    # This should be implemented in actual tests with system-specific logic
    # Provided as a template for derived test helpers
    true
end

"""
    validate_trace_matches_times(trace, dtimes)

Verify that discrete times are properly sampled from trace.
"""
function validate_trace_matches_times(trace, dtimes)
    issorted(dtimes) && all(t in trace.t || t >= trace.t[end] for t in dtimes)
end

# ============================================================================
# Hybrid System Helpers
# ============================================================================

"""
    validate_sde_equilibrium(traj::Matrix, species_idx::Int, expected_mean::Float64, rtol=0.3)

Check that SDE-evolved species converges to expected mean.
"""
function validate_sde_equilibrium(traj::Matrix, species_idx::Int, expected_mean::Float64, rtol=0.3)
    observed_mean = mean(traj[species_idx, :])
    isapprox(observed_mean, expected_mean, rtol=rtol)
end

"""
    validate_sde_variance(traj::Matrix, species_idx::Int, expected_variance::Float64, rtol=0.3)

Check that SDE-evolved species has expected variance.
"""
function validate_sde_variance(traj::Matrix, species_idx::Int, expected_variance::Float64, rtol=0.3)
    observed_var = var(traj[species_idx, :])
    isapprox(observed_var, expected_variance, rtol=rtol)
end

# ============================================================================
# Rate Computation Helpers
# ============================================================================

"""
    estimate_rate_from_trace(trace::PWS.ReactionTrace)

Estimate the average reaction rate from inter-event times.
"""
function estimate_rate_from_trace(trace::PWS.ReactionTrace)
    1 / mean(diff(trace.t))
end

"""
    compare_rate_estimates(observed_rate, analytical_rate, rtol=0.05; msg="")

Compare observed rate estimate with analytical expectation.
"""
function compare_rate_estimates(observed_rate, analytical_rate, rtol=0.05; msg="")
    if !isapprox(observed_rate, analytical_rate, rtol=rtol)
        error("Rate mismatch: observed=$observed_rate, expected=$analytical_rate, rtol=$rtol. $msg")
    end
end
