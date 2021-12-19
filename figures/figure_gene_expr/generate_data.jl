# This file contains the code to generate the PWS data used for
# the paper.

# WARNING: This script takes a long time (hours to days) to complete
# This file should be executed on multiple CPU's (e.g.
# on a computing cluster). To use all cores of the current
# computer, start julia with `julia -p auto generate_data.jl`.

using Logging

@info "Loading Packages"
using HDF5
@everywhere using PathWeightSampling
using DrWatson
using Distributions
using ProgressMeter

duration = 10
parameters = (;
    mean_s = 50,
    corr_time_s = 1.0,
    corr_time_x = 0.1,
    dtimes = collect(0.0:0.1:duration)
)

mean_s = parameters.mean_s
κ = parameters.mean_s / parameters.corr_time_s
λ = 1 / parameters.corr_time_s
ρ = 1 / parameters.corr_time_x
μ = 1 / parameters.corr_time_x

# Use a Gaussian approximation for the initial condition
@info "Compute Initial Condition"
covariance = [mean_s (ρ*mean_s / (λ + μ)); (ρ*mean_s / (λ + μ)) (mean_s * (1 + ρ/(λ + μ)))]
initial_dist = MvNormal([mean_s, mean_s], covariance)
s_axis = 25:75
x_axis = 25:75
z = pdf.(Ref(initial_dist), [[i,j] for i=s_axis, j=x_axis])
initial_condition = PathWeightSampling.empirical_dist(z, s_axis, x_axis)

function run_simulation(systemfn, algorithm, num_responses)
    @everywhere begin
        system = $systemfn()
        global compiled_system = PathWeightSampling.compile(system)
    end
    
    result = @showprogress pmap(1:num_responses) do batch
        PathWeightSampling._mi_inner(Main.compiled_system, algorithm, 1, false)
    end
    vcat(result...)
end

# This function runs a simulation and saves it in the data directory
function save_simulation(alg::Symbol, N::Integer, M::Integer, parameters)
    systemfn = () -> PathWeightSampling.gene_expression_system(; u0=initial_condition, parameters...)
    
    algorithm = if alg == :directmc
        DirectMCEstimate(M)
    elseif alg == :smc
        SMCEstimate(M)
    elseif alg == :ti
        TIEstimate(0, 8, M÷8)
    else
        error("No algorithm $alg found.")
    end
    
    mi = run_simulation(systemfn, algorithm, N)
    result = Dict("Samples" => mi, "Parameters" => Dict(pairs(parameters)))
    
    Alg = PathWeightSampling.name(algorithm)
    Duration = duration
    filename = savename((@dict Alg Duration M), "hdf5")
    local_path = datadir("gene_expression", filename)
    mkpath(dirname(local_path))
    h5open(local_path, "w") do file
        PathWeightSampling.write_hdf5!(file, result)
    end
end

# Reduce these numbers to make the computations complete
# faster
N = 10_000
M_vals = [128, 1024, 4096]

@info "Starting DPWS simulations"
for M in M_vals
    save_simulation(:directmc, N, M, parameters)
end

@info "Starting TI-PWS simulations"
for T in [1:10], M in M_vals
    parameters = (;
        mean_s = 50,
        corr_time_s = 1.0,
        corr_time_x = 0.1,
        dtimes = collect(0.0:0.1:T)
    )
    save_simulation(:ti, N, M, parameters)
end

@info "Starting RR-PWS simulations"
for M in [128]
    save_simulation(:smc, N, M, parameters)
end