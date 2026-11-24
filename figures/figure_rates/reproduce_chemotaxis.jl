#!/usr/bin/env julia
#
# Recompute the data behind the chemotaxis figure (`figure_chemotaxis.jl`) with PathWeightSampling.jl.
#
# The figure shows, for the literature chemotaxis model (400 receptor clusters, 6 methylation
# levels), the mutual information I(c; Yp) between the ligand trajectory c and the CheY-P
# trajectory over a trajectory duration of 20 s, for four gradient steepnesses, together with a
# summary of simulated input signals. This script regenerates the two data files the figure reads:
#
#     2023-02-17_mi_trajectories.csv   mean MI (nats) over N trajectories vs trajectory duration
#     input_traj_summary.csv           quantiles and samples of the input signal c(t)
#
# Run it from the repository root with the *package* environment (not `figures/`):
#
#     julia --project=. -t auto figures/figure_rates/reproduce_chemotaxis.jl compute
#     julia --project=. figures/figure_rates/reproduce_chemotaxis.jl assemble
#     julia --project=. figures/figure_rates/reproduce_chemotaxis.jl input
#     julia --project=. figures/figure_rates/reproduce_chemotaxis.jl check
#
# then plot with the `figures/` environment, see README.md in this directory.
#
# Commands
#   compute    estimate the MI of N trajectories per gradient steepness (the expensive part).
#              Results are written per gradient and chunk to <out>/partial/, finished pieces are
#              skipped on a restart, and `--chunk i/n` computes only the i-th of n slices of the
#              trajectories so that the work can be spread over many jobs.
#   assemble   merge all chunks into <out>/mi_trajectories.csv
#   input      simulate input signals and write <out>/input_traj_summary.csv
#   check      compare <out>/mi_trajectories.csv with the published data
#
# Options (`--name value` or `--name=value`); the defaults are the parameters of the paper
#   --clusters 400          number of receptor clusters
#   --lmax 6                number of methylation sites per receptor (Lmax; `n` in the code)
#   --duration 20           trajectory duration in s
#   --dt 0.1                output time step in s
#   --gradients 1e-4,2e-4,3e-4,4e-4   gradient steepness g (1/µm)
#   --ntraj 14400           trajectories per gradient steepness
#   --particles 256         SMC particles per marginalization
#   --algorithm aggregated  `aggregated` (AggregatedSMCEstimate) or `generic` (SMCEstimate)
#   --chunk 1/1             slice of the trajectories to compute
#   --input-trajectories 1000   trajectories for the input-signal summary
#   --out figures/figure_rates/data/reproduced
#
# The seed of every trajectory is a function of (gradient, trajectory index) only, so the result
# does not depend on the number of threads, chunks or the order of execution.

import PathWeightSampling as PWS
import Random: Xoshiro
using Printf
using Statistics
using StochasticDiffEq

const HERE = @__DIR__

Base.@kwdef struct Options
    clusters::Int = 400
    lmax::Int = 6
    duration::Float64 = 20.0
    dt::Float64 = 0.1
    gradients::Vector{Float64} = [1e-4, 2e-4, 3e-4, 4e-4]
    ntraj::Int = 14400
    particles::Int = 256
    algorithm::String = "aggregated"
    chunk::Tuple{Int,Int} = (1, 1)
    input_trajectories::Int = 1000
    out::String = joinpath(HERE, "data", "reproduced")
end

function parse_options(args)
    command = nothing
    kw = Dict{Symbol,Any}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            name, value = occursin('=', arg) ? split(arg[3:end], '=', limit=2) : (arg[3:end], nothing)
            if value === nothing
                i += 1
                i <= length(args) || error("missing value for $arg")
                value = args[i]
            end
            name = replace(name, '-' => '_')
            if name == "gradients"
                kw[:gradients] = parse.(Float64, split(value, ','))
            elseif name == "chunk"
                k, n = parse.(Int, split(value, '/'))
                1 <= k <= n || error("--chunk i/n needs 1 <= i <= n")
                kw[:chunk] = (k, n)
            elseif name in ("clusters", "lmax", "ntraj", "particles", "input_trajectories")
                kw[Symbol(name)] = parse(Int, value)
            elseif name in ("duration", "dt")
                kw[Symbol(name)] = parse(Float64, value)
            elseif name in ("algorithm", "out")
                kw[Symbol(name)] = String(value)
            else
                error("unknown option --$name")
            end
        elseif command === nothing
            command = arg
        else
            error("unexpected argument $arg")
        end
        i += 1
    end
    command, Options(; kw...)
end

# ---------------------------------------------------------------------------------------------
# Systems, algorithms, seeds

chemotaxis(o::Options, g; duration=o.duration) =
    PWS.chemotaxis_system(n_clusters=o.clusters, n=o.lmax, duration=duration, dt=o.dt, gradient_steepness=g)

function algorithm(o::Options)
    o.algorithm == "aggregated" && return PWS.AggregatedSMCEstimate(o.particles)
    o.algorithm == "generic" && return PWS.SMCEstimate(o.particles)
    error("unknown --algorithm $(o.algorithm) (use `aggregated` or `generic`)")
end

gtag(g) = @sprintf("%.6g", g)
trajectory_seed(g, i) = 100_000_000 * round(Int, 1e6 * g) + i  # one stream per (g, i)

partial_dir(o) = joinpath(o.out, "partial")
partial_file(o, g) = joinpath(partial_dir(o), "g$(gtag(g))_chunk$(o.chunk[1])of$(o.chunk[2]).csv")

function chunk_range(ntraj, (k, n))
    first = (k - 1) * ntraj ÷ n + 1
    last = k * ntraj ÷ n
    first:last
end

"""MI(t) in nats of one trajectory: ln P(y|c) - ln P(y) on the output time grid."""
function trajectory_information(system, alg, g, i)
    seed = trajectory_seed(g, i)
    conf = PWS.generate_configuration(system; rng=Xoshiro(2seed))
    mi, _ = PWS.information_density(system, alg, conf; rng=Xoshiro(2seed + 1))
    any(ismissing, mi) && error("non-finite MI estimate for g=$g, trajectory $i")
    Float64.(mi)
end

# ---------------------------------------------------------------------------------------------
# compute

function compute(o::Options)
    alg = algorithm(o)
    range = chunk_range(o.ntraj, o.chunk)
    mkpath(partial_dir(o))
    @info "compute" algorithm = PWS.name(alg) clusters = o.clusters lmax = o.lmax ntraj = o.ntraj chunk = o.chunk threads = Threads.nthreads()
    for g in o.gradients
        file = partial_file(o, g)
        if isfile(file)
            @info "skipping finished piece" file
            continue
        end
        base = chemotaxis(o, g)
        times = collect(PWS.discrete_times(base))
        nthreads = Threads.nthreads()
        sums = [zeros(length(times)) for _ in 1:nthreads]
        squares = [zeros(length(times)) for _ in 1:nthreads]
        next = Threads.Atomic{Int}(first(range))
        finished = Threads.Atomic{Int}(0)
        report_every = max(1, length(range) ÷ 20)
        started = time()
        @sync for t in 1:nthreads
            Threads.@spawn begin
                system = copy(base)  # the jump-rate aggregator is mutable state
                while true
                    i = Threads.atomic_add!(next, 1)
                    i > last(range) && break
                    mi = trajectory_information(system, alg, g, i)
                    sums[t] .+= mi
                    squares[t] .+= mi .^ 2
                    done = Threads.atomic_add!(finished, 1) + 1
                    done % report_every == 0 &&
                        @info "progress" g done total = length(range) elapsed_min = round((time() - started) / 60, digits=1)
                end
            end
        end
        total, total2 = sum(sums), sum(squares)
        tmp = file * ".tmp"
        open(tmp, "w") do io
            println(io, "DiscreteTimes,N,Sum,SumSq")
            for k in eachindex(times)
                @printf(io, "%.10g,%d,%.17g,%.17g\n", times[k], length(range), total[k], total2[k])
            end
        end
        mv(tmp, file)  # a finished piece is never half written
        @info "wrote" file mean_final_bits = total[end] / length(range) / log(2)
    end
end

# ---------------------------------------------------------------------------------------------
# assemble

function read_partial(file)
    lines = readlines(file)
    cols = [split(l, ',') for l in lines[2:end]]
    (; times=[parse(Float64, c[1]) for c in cols], n=parse(Int, cols[1][2]),
        sum=[parse(Float64, c[3]) for c in cols], sumsq=[parse(Float64, c[4]) for c in cols])
end

function assemble(o::Options)
    isdir(partial_dir(o)) || error("no data in $(partial_dir(o)); run `compute` first")
    system = chemotaxis(o, o.gradients[1])
    tau_l = 1 / system.sde_prob.p[1]  # input correlation time = 1/decay rate of the velocity
    outfile = joinpath(o.out, "mi_trajectories.csv")
    open(outfile, "w") do io
        println(io, "\"TauL\",\"DiscreteTimes\",\"Lmax\",\"NClusters\",\"g\",\"Mean\",\"Var\",\"M\",\"N\"")
        for g in o.gradients
            files = filter(f -> startswith(f, "g$(gtag(g))_chunk"), readdir(partial_dir(o)))
            isempty(files) && error("no pieces found for g=$g")
            nchunks = unique(parse(Int, match(r"of(\d+)\.csv$", f).captures[1]) for f in files)
            length(nchunks) == 1 || error("pieces for g=$g come from different --chunk splits: $files")
            length(files) == only(nchunks) || error("g=$g: found $(length(files)) of $(only(nchunks)) pieces")
            parts = [read_partial(joinpath(partial_dir(o), f)) for f in files]
            n = sum(p.n for p in parts)
            n == o.ntraj || @warn "g=$g: $n trajectories instead of --ntraj $(o.ntraj)"
            total = sum(p.sum for p in parts)
            total2 = sum(p.sumsq for p in parts)
            mean_mi = total ./ n
            variance_of_mean = max.(total2 ./ n .- mean_mi .^ 2, 0) ./ (n - 1)  # sample variance / N
            for k in eachindex(mean_mi)
                @printf(io, "%.15g,%.10g,%d,%d,%.6g,%.15g,%.15g,%d,%d\n", tau_l, parts[1].times[k],
                    o.lmax, o.clusters, g, mean_mi[k], variance_of_mean[k], o.particles, n)
            end
        end
    end
    @info "wrote" outfile
end

# ---------------------------------------------------------------------------------------------
# input

function input_summary(o::Options)
    g = 2e-4  # the gradient steepness used for the input-signal panel
    duration = 100.0
    system = chemotaxis(o, g; duration)
    ts = collect(range(0.0, duration, step=0.1))
    # the ligand is an exact function of the SDE state: integrate the SDE of the system
    signals = Matrix{Float64}(undef, length(ts), o.input_trajectories)
    for j in 1:o.input_trajectories
        sol = solve(system.sde_prob, EM(); dt=system.sde_dt, saveat=0.1, seed=trajectory_seed(g, j))
        signals[:, j] = [u[2] for u in sol.u]
    end
    mkpath(o.out)
    outfile = joinpath(o.out, "input_traj_summary.csv")
    nsample = min(20, o.input_trajectories)
    open(outfile, "w") do io
        println(io, join(["t", "mean", "q05", "q95", "q25", "q75", ("traj$j" for j in 1:nsample)...], ','))
        for (k, t) in enumerate(ts)
            row = @view signals[k, :]
            vals = [t, mean(row), quantile(row, 0.05), quantile(row, 0.95), quantile(row, 0.25),
                quantile(row, 0.75), (signals[k, j] for j in 1:nsample)...]
            println(io, join((@sprintf("%.15g", v) for v in vals), ','))
        end
    end
    @info "wrote" outfile
end

# ---------------------------------------------------------------------------------------------
# check

function read_table(file)
    lines = readlines(file)
    header = strip.(split(lines[1], ','), '"')
    [Dict(zip(header, strip.(split(l, ','), '"'))) for l in lines[2:end]]
end

function check(o::Options)
    mine = read_table(joinpath(o.out, "mi_trajectories.csv"))
    published = read_table(joinpath(HERE, "data", "2023-02-17_mi_trajectories.csv"))
    final(rows, g) = last(filter(r -> isapprox(parse(Float64, r["g"]), g, rtol=1e-6), rows))
    println("MI at the end of the trajectory in bits (published: M=256, N=14400; ± is the s.e. of the mean)")
    println(rpad("g (1/mm)", 10), rpad("T (s)", 8), rpad("published", 18), rpad("reproduced", 18), "difference / s.e.")
    for g in o.gradients
        pub, rep = final(published, g), final(mine, g)
        f(r) = (parse(Float64, r["Mean"]) / log(2), sqrt(parse(Float64, r["Var"])) / log(2))
        (pm, ps), (rm, rs) = f(pub), f(rep)
        z = (rm - pm) / hypot(ps, rs)
        println(rpad(round(g * 1e3, digits=3), 10), rpad(parse(Float64, rep["DiscreteTimes"]), 8),
            rpad(@sprintf("%.4f ± %.4f", pm, ps), 18), rpad(@sprintf("%.4f ± %.4f", rm, rs), 18), @sprintf("%+.1f", z))
    end
end

# ---------------------------------------------------------------------------------------------

function main(args)
    command, o = parse_options(args)
    commands = Dict("compute" => compute, "assemble" => assemble, "input" => input_summary, "check" => check)
    haskey(commands, command) || error("usage: reproduce_chemotaxis.jl {$(join(keys(commands), "|"))} [options]; see the header of this file")
    commands[command](o)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main(ARGS)
end
