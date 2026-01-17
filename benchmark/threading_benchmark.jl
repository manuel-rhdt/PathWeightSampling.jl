#!/usr/bin/env julia
"""
    threading_benchmark.jl

Benchmark script to test multithreaded performance of PathWeightSampling.
Creates a simple birth-death gene expression system and compares
single-threaded vs multithreaded mutual information computation.

Usage:
    julia --project=. --threads=auto benchmark/threading_benchmark.jl
    julia --project=. --threads=4 benchmark/threading_benchmark.jl
"""

using PathWeightSampling
import PathWeightSampling as PWS
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

# Summary
println("=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println()
println("Timing:")
@printf("  Single-threaded time: %.3f s\n", t_single)
@printf("  Multi-threaded time:  %.3f s\n", t_multi)
@printf("  Speedup: %.2fx\n", t_single / t_multi)
@printf("  Parallel efficiency: %.1f%%\n", 100 * (t_single / t_multi) / Threads.nthreads())
println()

println("Memory & GC:")
@printf("  Single-threaded allocations: %s\n", format_bytes(gc_single.allocd))
@printf("  Multi-threaded allocations:  %s\n", format_bytes(gc_multi.allocd))
@printf("  Allocation ratio (multi/single): %.2fx\n", gc_multi.allocd / gc_single.allocd)
@printf("  Single-threaded GC time: %.1f ms (%.1f%% of runtime)\n", gc_single.total_time_ns / 1e6, 100 * gc_single.total_time_ns / 1e9 / t_single)
@printf("  Multi-threaded GC time:  %.1f ms (%.1f%% of runtime)\n", gc_multi.total_time_ns / 1e6, 100 * gc_multi.total_time_ns / 1e9 / t_multi)
println()

# Verify results are consistent
mi_diff = abs(mi_single - mi_multi) / mi_single * 100
println("Correctness:")
@printf("  MI difference: %.2f%% (should be small, due to Monte Carlo variance)\n", mi_diff)

if t_single / t_multi > 1.5 && Threads.nthreads() > 1
    println("\n✓ Multi-threading is providing speedup!")
elseif Threads.nthreads() == 1
    println("\n⚠ Running with only 1 thread. Use --threads=auto for parallelism.")
else
    println("\n⚠ Limited speedup observed. This may be normal for small workloads.")
end
println()
