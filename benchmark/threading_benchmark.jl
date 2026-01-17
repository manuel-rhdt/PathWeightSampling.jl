#!/usr/bin/env julia
"""
    threading_benchmark.jl

Benchmark script to test multithreaded and distributed performance of PathWeightSampling.
Creates a simple birth-death gene expression system and compares
single-threaded, multithreaded, and distributed mutual information computation.

Usage:
    julia --project=. --threads=auto benchmark/threading_benchmark.jl
    julia --project=. --threads=4 benchmark/threading_benchmark.jl
"""

using Distributed

# Add worker processes (same number as threads for fair comparison)
num_workers = Threads.nthreads()
if nworkers() < num_workers
    addprocs(num_workers - nworkers() + 1)
end

@everywhere using PathWeightSampling
@everywhere import PathWeightSampling as PWS

using Statistics
using Printf

# Helper function to format bytes
function format_bytes(bytes)
    if bytes < 1024
        return @sprintf("%d B", bytes)
    elseif bytes < 1024^2
        return @sprintf("%.2f KiB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%.2f MiB", bytes / 1024^2)
    else
        return @sprintf("%.2f GiB", bytes / 1024^3)
    end
end

# Helper to compute GC stats difference
function gc_diff(after::Base.GC_Num, before::Base.GC_Num)
    return (
        allocd = after.allocd - before.allocd,
        total_time_ns = after.total_time - before.total_time,
        pause = after.pause - before.pause,
        full_sweep = after.full_sweep - before.full_sweep,
    )
end

# Print system info
println("=" ^ 60)
println("PathWeightSampling Threading Benchmark")
println("=" ^ 60)
println()
println("System Information:")
println("  Julia version: $(VERSION)")
println("  Number of threads: $(Threads.nthreads())")
println("  Number of workers: $(nworkers())")
println("  Number of CPUs: $(Sys.CPU_THREADS)")
println()

# Create a simple gene expression system
println("Setting up gene expression system...")
system = PWS.gene_expression_system(
    dtimes=0:0.1:10.0
)
println("  Duration: $(system.tspan[2]) time units")
println("  Time step: 0.1")
println("  Number of time points: $(length(PWS.discrete_times(system)))")
println()

# Benchmark parameters
num_samples = 1000
num_particles = 128
num_warmup = 2  # Warmup runs to trigger compilation

println("Benchmark Parameters:")
println("  Number of samples: $num_samples")
println("  Number of particles: $num_particles")
println("  Algorithm: SMCEstimate")
println()

# Warmup (compile code paths)
println("Warming up (compiling code paths)...")
for _ in 1:num_warmup
    PWS.mutual_information(system, PWS.SMCEstimate(32), num_samples=10, threads=false, progress=false)
    PWS.mutual_information(system, PWS.SMCEstimate(32), num_samples=10, threads=true, progress=false)
end
# Warmup distributed (compile on workers)
PWS.mutual_information(system, PWS.SMCEstimate(32), num_samples=nworkers()*2, distributed=true, progress=false)
println("  Warmup complete.")
println()

# Single-threaded benchmark
println("-" ^ 60)
println("Running single-threaded benchmark...")
GC.gc()
gc_before_single = Base.gc_num()
t_single_start = time()
result_single = PWS.mutual_information(
    system,
    PWS.SMCEstimate(num_particles),
    num_samples=num_samples,
    threads=false,
    progress=true
)
t_single = time() - t_single_start
gc_after_single = Base.gc_num()
gc_single = gc_diff(gc_after_single, gc_before_single)

df_single = result_single.result
mi_single = mean(df_single[df_single.time .== maximum(df_single.time), :MutualInformation])

println()
println("Single-threaded results:")
@printf("  Time: %.3f seconds\n", t_single)
@printf("  Throughput: %.1f samples/second\n", num_samples / t_single)
@printf("  Final MI: %.4f nats\n", mi_single)
println("  GC stats:")
@printf("    Allocated: %s\n", format_bytes(gc_single.allocd))
@printf("    GC time: %.3f ms (%.1f%% of total)\n", gc_single.total_time_ns / 1e6, 100 * gc_single.total_time_ns / 1e9 / t_single)
@printf("    GC pauses: %d\n", gc_single.pause)
@printf("    Full sweeps: %d\n", gc_single.full_sweep)
println()

# Multi-threaded benchmark
println("-" ^ 60)
println("Running multi-threaded benchmark ($(Threads.nthreads()) threads)...")
GC.gc()
gc_before_multi = Base.gc_num()
t_multi_start = time()
result_multi = PWS.mutual_information(
    system,
    PWS.SMCEstimate(num_particles),
    num_samples=num_samples,
    threads=true,
    progress=true
)
t_multi = time() - t_multi_start
gc_after_multi = Base.gc_num()
gc_multi = gc_diff(gc_after_multi, gc_before_multi)

df_multi = result_multi.result
mi_multi = mean(df_multi[df_multi.time .== maximum(df_multi.time), :MutualInformation])

println()
println("Multi-threaded results:")
@printf("  Time: %.3f seconds\n", t_multi)
@printf("  Throughput: %.1f samples/second\n", num_samples / t_multi)
@printf("  Final MI: %.4f nats\n", mi_multi)
println("  GC stats:")
@printf("    Allocated: %s\n", format_bytes(gc_multi.allocd))
@printf("    GC time: %.3f ms (%.1f%% of total)\n", gc_multi.total_time_ns / 1e6, 100 * gc_multi.total_time_ns / 1e9 / t_multi)
@printf("    GC pauses: %d\n", gc_multi.pause)
@printf("    Full sweeps: %d\n", gc_multi.full_sweep)
println()

# Distributed benchmark
println("-" ^ 60)
println("Running distributed benchmark ($(nworkers()) workers)...")
GC.gc()
gc_before_dist = Base.gc_num()
t_dist_start = time()
result_dist = PWS.mutual_information(
    system,
    PWS.SMCEstimate(num_particles),
    num_samples=num_samples,
    distributed=true,
    progress=true
)
t_dist = time() - t_dist_start
gc_after_dist = Base.gc_num()
gc_dist = gc_diff(gc_after_dist, gc_before_dist)

df_dist = result_dist.result
mi_dist = mean(df_dist[df_dist.time .== maximum(df_dist.time), :MutualInformation])

println()
println("Distributed results:")
@printf("  Time: %.3f seconds\n", t_dist)
@printf("  Throughput: %.1f samples/second\n", num_samples / t_dist)
@printf("  Final MI: %.4f nats\n", mi_dist)
println("  GC stats (main process only):")
@printf("    Allocated: %s\n", format_bytes(gc_dist.allocd))
@printf("    GC time: %.3f ms (%.1f%% of total)\n", gc_dist.total_time_ns / 1e6, 100 * gc_dist.total_time_ns / 1e9 / t_dist)
@printf("    GC pauses: %d\n", gc_dist.pause)
@printf("    Full sweeps: %d\n", gc_dist.full_sweep)
println()

# Summary
println("=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println()
println("Timing:")
@printf("  Single-threaded time: %.3f s\n", t_single)
@printf("  Multi-threaded time:  %.3f s (%.2fx speedup, %.1f%% efficiency)\n",
    t_multi, t_single / t_multi, 100 * (t_single / t_multi) / Threads.nthreads())
@printf("  Distributed time:     %.3f s (%.2fx speedup, %.1f%% efficiency)\n",
    t_dist, t_single / t_dist, 100 * (t_single / t_dist) / nworkers())
println()

println("Throughput:")
@printf("  Single-threaded: %.1f samples/s\n", num_samples / t_single)
@printf("  Multi-threaded:  %.1f samples/s\n", num_samples / t_multi)
@printf("  Distributed:     %.1f samples/s\n", num_samples / t_dist)
println()

println("Memory & GC (main process):")
@printf("  Single-threaded allocations: %s\n", format_bytes(gc_single.allocd))
@printf("  Multi-threaded allocations:  %s (%.2fx)\n", format_bytes(gc_multi.allocd), gc_multi.allocd / gc_single.allocd)
@printf("  Distributed allocations:     %s (%.2fx)\n", format_bytes(gc_dist.allocd), gc_dist.allocd / gc_single.allocd)
@printf("  Single-threaded GC time: %.1f ms (%.1f%% of runtime)\n", gc_single.total_time_ns / 1e6, 100 * gc_single.total_time_ns / 1e9 / t_single)
@printf("  Multi-threaded GC time:  %.1f ms (%.1f%% of runtime)\n", gc_multi.total_time_ns / 1e6, 100 * gc_multi.total_time_ns / 1e9 / t_multi)
@printf("  Distributed GC time:     %.1f ms (%.1f%% of runtime)\n", gc_dist.total_time_ns / 1e6, 100 * gc_dist.total_time_ns / 1e9 / t_dist)
println()

# Verify results are consistent
mi_diff_multi = abs(mi_single - mi_multi) / mi_single * 100
mi_diff_dist = abs(mi_single - mi_dist) / mi_single * 100
println("Correctness (MI differences, should be small due to Monte Carlo variance):")
@printf("  Multi-threaded vs single: %.2f%%\n", mi_diff_multi)
@printf("  Distributed vs single:    %.2f%%\n", mi_diff_dist)

println()
println("Best parallelization strategy:")
if t_multi <= t_dist
    @printf("  Multi-threading is faster (%.2fx vs %.2fx speedup)\n", t_single / t_multi, t_single / t_dist)
else
    @printf("  Distributed is faster (%.2fx vs %.2fx speedup)\n", t_single / t_dist, t_single / t_multi)
end
println()
