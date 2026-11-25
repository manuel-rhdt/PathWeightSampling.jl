#!/usr/bin/env julia
#
# Recompute the noise autocovariance and the response kernel of the chemotaxis model, the data
# behind `figure_kernels.jl` and the comparison figure in `figure_rates/`:
#
#     cov_data_lit.csv, cov_data_fit.csv      autocovariance of the kinase activity
#     kern_data_lit.csv, kern_data_fit.csv    response of the activity to a step in the ligand concentration
#
# "lit" is the literature parameter set (400 clusters with 6 methylation levels), "fit" the fitted one
# (9 clusters with 15 levels). The data was originally produced in a notebook with an older version
# of this package (see figures/figure_rates/PROVENANCE.md); this script runs the same computation with
# the current package. It simulates the full jump process of the receptors and CheY, no PWS estimate is
# involved.
#
# Run it from the repository root with the *package* environment (not `figures/`):
#
#     julia --project=. -t auto figures/figure_kernels/simulate_kernels.jl noise
#     julia --project=. -t auto figures/figure_kernels/simulate_kernels.jl kernel
#     julia --project=. figures/figure_kernels/simulate_kernels.jl assemble
#     julia --project=. figures/figure_kernels/simulate_kernels.jl check
#
# The Fourier-space kernels and the Gaussian information rates are computed from the assembled files
# with `gaussian_rates.jl` (in the `figures/` environment).
#
# Commands
#   noise      simulate trajectories of 1000 s at constant ligand concentration (100 µM) and estimate the
#              autocovariance of the kinase activity a = (Yp/Y) k_Z / (k_A n_clusters), obtained from the
#              CheY phosphorylation state as in experiments
#   kernel     simulate trajectories in which the ligand concentration steps from 100 to 110 µM at t = 50 s
#              and estimate the response kernel K(t) = -(a(t) - a(0)) / ln(110/100)
#   assemble   merge the pieces written by the two commands above into the CSV files
#   check      compare the assembled files with the committed data
#
# Both commands write one piece per model and chunk to <out>/partial/; finished pieces are skipped when
# a command is restarted and `--chunk i/n` computes only the i-th of n slices of the trajectories, so the
# work can be spread over many jobs. The seed of each trajectory depends only on (command, model, index).
#
# Options (`--name value` or `--name=value`); the defaults are the parameters of the original computation
#   --models lit,fit    parameter sets
#   --ntraj             trajectories per model: 10000 for `noise`, 100000 for `kernel`
#   --chunk 1/1         slice of the trajectories to compute
#   --out figures/figure_kernels/data/reproduced

import PathWeightSampling as PWS
import Random: Xoshiro
using Printf
using Statistics
using StatsBase: autocov

const HERE = @__DIR__

const MODELS = Dict("lit" => (clusters=400, lmax=6), "fit" => (clusters=9, lmax=15))

# the experiment of the response kernel: ligand step from `before` to `after` µM at `time`
const STEP = (time=50.0, before=100.0, after=110.0)

# the simulations of the two commands
const SETTINGS = Dict(
    "noise" => (duration=1000.0, dt=0.01, maxlag=20_000, default_ntraj=10_000),
    "kernel" => (duration=250.0, dt=0.01, maxlag=0, default_ntraj=100_000),
)

Base.@kwdef struct Options
    models::Vector{String} = ["lit", "fit"]
    ntraj::Union{Nothing,Int} = nothing
    chunk::Tuple{Int,Int} = (1, 1)
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
            if name == "models"
                models = String.(split(value, ','))
                all(m -> haskey(MODELS, m), models) || error("unknown model in --models $value (use lit, fit)")
                kw[:models] = models
            elseif name == "chunk"
                k, n = parse.(Int, split(value, '/'))
                1 <= k <= n || error("--chunk i/n needs 1 <= i <= n")
                kw[:chunk] = (k, n)
            elseif name == "ntraj"
                kw[:ntraj] = parse(Int, value)
            elseif name == "out"
                kw[:out] = String(value)
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
# Simulation

system_for(model, kind) = PWS.chemotaxis_system(
    n_clusters=MODELS[model].clusters, n=MODELS[model].lmax,
    duration=SETTINGS[kind].duration, dt=SETTINGS[kind].dt, sde_dt=SETTINGS[kind].dt)

trajectory_seed(kind, model, i) = ((kind == "noise" ? 1 : 2) * 10 + (model == "lit" ? 1 : 2)) * 1_000_000_000 + i

"""CheY (Y) and phosphorylated CheY (Yp) on the output grid for one trajectory of the full jump process
at constant ligand concentration. With `step`, the ligand is set to `step.after` at `step.time`."""
function simulate(system, rng; step=nothing)
    j = system.reactions
    all = BitSet(1:PWS.SSA.num_reactions(j))
    agg = PWS.SSA.initialize_aggregator!(system.agg, j; u0=copy(system.u0), traced_reactions=BitSet(),
        active_reactions=all, tspan=system.tspan, seed=rand(rng, UInt32))
    times = collect(PWS.discrete_times(system))
    Y = Vector{Float64}(undef, length(times))
    Yp = similar(Y)
    Y[1], Yp[1] = agg.u[j.Y], agg.u[j.Yp]
    stepped = step === nothing
    for i in 2:length(times)
        PWS.JumpSystem.advance_ssa!(agg, j, times[i], nothing, nothing)
        if !stepped && times[i] >= step.time
            agg.u[j.ligand] = step.after
            PWS.SSA.update_rates!(agg, j)
            stepped = true
        end
        Y[i], Yp[i] = agg.u[j.Y], agg.u[j.Yp]
    end
    times, Y, Yp
end

n_clusters(system) = round(Int, sum(system.u0[system.reactions.receptors]))

# kinase activity from the CheY phosphorylation state: a = (Yp / Y) k_Z / (k_A n_clusters)
activity_scale(system) = system.reactions.k_Z / (system.reactions.k_A * n_clusters(system))

"""Autocovariance of the activity of one trajectory for lags 0 … maxlag steps (the `noise` statistic)."""
function noise_statistic(system, rng, maxlag)
    times, Y, Yp = simulate(system, rng)
    a = Yp ./ Y .* activity_scale(system)
    autocov(a, 0:maxlag)
end

"""Response kernel of one trajectory for t ≥ 0 after the ligand step (the `kernel` statistic)."""
function kernel_statistic(system, rng)
    times, Y, Yp = simulate(system, rng; step=STEP)
    i0 = searchsortedfirst(times, STEP.time)
    y, yp = Y[i0:end], Yp[i0:end]
    kernel = -(yp ./ y .- yp[1] / y[1]) ./ log(STEP.after / STEP.before) .* activity_scale(system)
    kernel
end

# ---------------------------------------------------------------------------------------------
# Running statistics (Welford): count, mean and sum of squared deviations per time point

mutable struct Accumulator
    n::Int
    mean::Vector{Float64}
    m2::Vector{Float64}
end
Accumulator(len) = Accumulator(0, zeros(len), zeros(len))

function add!(a::Accumulator, x)
    a.n += 1
    @. a.m2 += (x - a.mean) * (x - (a.mean + (x - a.mean) / a.n))
    @. a.mean += (x - a.mean) / a.n
    a
end

function merge(a::Accumulator, b::Accumulator)
    a.n == 0 && return b
    b.n == 0 && return a
    n = a.n + b.n
    Accumulator(n, (a.n .* a.mean .+ b.n .* b.mean) ./ n,
        @. a.m2 + b.m2 + (a.mean - b.mean)^2 * a.n * b.n / n)
end

# ---------------------------------------------------------------------------------------------
# noise and kernel

partial_dir(o) = joinpath(o.out, "partial")
partial_file(o, kind, model) = joinpath(partial_dir(o), "$(kind)_$(model)_chunk$(o.chunk[1])of$(o.chunk[2]).csv")

ntraj(o, kind) = something(o.ntraj, SETTINGS[kind].default_ntraj)

function chunk_range(n, (k, nchunks))
    ((k - 1) * n ÷ nchunks + 1):(k * n ÷ nchunks)
end

function run_statistic(o::Options, kind)
    mkpath(partial_dir(o))
    range = chunk_range(ntraj(o, kind), o.chunk)
    @info "simulate" kind models = o.models ntraj = ntraj(o, kind) chunk = o.chunk threads = Threads.nthreads()
    for model in o.models
        file = partial_file(o, kind, model)
        if isfile(file)
            @info "skipping finished piece" file
            continue
        end
        base = system_for(model, kind)
        len = SETTINGS[kind].maxlag > 0 ? SETTINGS[kind].maxlag + 1 : round(Int, (SETTINGS[kind].duration - STEP.time) / SETTINGS[kind].dt) + 1
        accs = [Accumulator(len) for _ in 1:Threads.nthreads()]
        next = Threads.Atomic{Int}(first(range))
        finished = Threads.Atomic{Int}(0)
        report_every = max(1, length(range) ÷ 20)
        started = time()
        @sync for t in 1:Threads.nthreads()
            Threads.@spawn begin
                system = copy(base)  # the jump-rate aggregator is mutable state
                while true
                    i = Threads.atomic_add!(next, 1)
                    i > last(range) && break
                    rng = Xoshiro(trajectory_seed(kind, model, i))
                    x = kind == "noise" ? noise_statistic(system, rng, SETTINGS[kind].maxlag) : kernel_statistic(system, rng)
                    add!(accs[t], x)
                    done = Threads.atomic_add!(finished, 1) + 1
                    done % report_every == 0 &&
                        @info "progress" kind model done total = length(range) elapsed_min = round((time() - started) / 60, digits=1)
                end
            end
        end
        acc = reduce(merge, accs)
        time_axis = (0:(len-1)) .* SETTINGS[kind].dt
        tmp = file * ".tmp"
        open(tmp, "w") do io
            println(io, "t,N,Mean,SumSqDev")
            for k in eachindex(time_axis)
                @printf(io, "%.10g,%d,%.17g,%.17g\n", time_axis[k], acc.n, acc.mean[k], acc.m2[k])
            end
        end
        mv(tmp, file)
        @info "wrote" file
    end
end

# ---------------------------------------------------------------------------------------------
# assemble and check

function read_partial(file)
    cols = [split(l, ',') for l in readlines(file)[2:end]]
    n = parse(Int, cols[1][2])
    t = [parse(Float64, c[1]) for c in cols]
    t, Accumulator(n, [parse(Float64, c[3]) for c in cols], [parse(Float64, c[4]) for c in cols])
end

function assemble(o::Options)
    for (kind, name, column) in (("noise", "cov_data", "Cov"), ("kernel", "kern_data", "Kernel"))
        for model in o.models
            files = filter(f -> startswith(f, "$(kind)_$(model)_chunk"), isdir(partial_dir(o)) ? readdir(partial_dir(o)) : String[])
            isempty(files) && (@info "no $kind pieces for $model, skipping"; continue)
            nchunks = unique(parse(Int, match(r"of(\d+)\.csv$", f).captures[1]) for f in files)
            length(nchunks) == 1 || error("$kind pieces for $model come from different --chunk splits: $files")
            length(files) == only(nchunks) || error("$kind, $model: found $(length(files)) of $(only(nchunks)) pieces")
            parts = [read_partial(joinpath(partial_dir(o), f)) for f in files]
            t = parts[1][1]
            acc = reduce(merge, (p[2] for p in parts))
            acc.n == ntraj(o, kind) || @warn "$kind, $model: $(acc.n) trajectories instead of $(ntraj(o, kind))"
            outfile = joinpath(o.out, "$(name)_$(model).csv")
            open(outfile, "w") do io
                println(io, "\"t\",\"$column\",\"SqDiff\",\"Var\",\"Nobs\"")
                for k in eachindex(t)
                    @printf(io, "%.10g,%.17g,%.17g,%.17g,%d\n", t[k], acc.mean[k], acc.m2[k], acc.m2[k] / (acc.n - 1), acc.n)
                end
            end
            @info "wrote" outfile
        end
    end
end

function read_table(file)
    lines = readlines(file)
    header = strip.(split(lines[1], ','), '"')
    cols = [parse.(Float64, strip.(getindex.(split.(lines[2:end], ','), j), '"')) for j in eachindex(header)]
    Dict(zip(header, cols))
end

function check(o::Options)
    println("Mean of the estimates of the new and the committed data, with the difference in units of the combined standard error.")
    println("(Var is the variance of a single trajectory, the standard error of the mean is sqrt(Var / Nobs).)")
    for (name, column) in (("cov_data", "Cov"), ("kern_data", "Kernel")), model in o.models
        mine_file = joinpath(o.out, "$(name)_$(model).csv")
        isfile(mine_file) || continue
        mine = read_table(mine_file)
        pub = read_table(joinpath(HERE, "data", "$(name)_$(model).csv"))
        println("\n$name $model: $(Int(mine["Nobs"][1])) new vs $(Int(pub["Nobs"][1])) committed trajectories")
        println(rpad("t (s)", 8), rpad("new", 14), rpad("committed", 14), "difference / s.e.")
        z = Float64[]
        for k in eachindex(mine["t"])
            se = sqrt(mine["Var"][k] / mine["Nobs"][k] + pub["Var"][k] / pub["Nobs"][k])
            se > 0 && push!(z, (mine[column][k] - pub[column][k]) / se)
        end
        for t in (0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0)
            k = findfirst(x -> isapprox(x, t, atol=1e-9), mine["t"])
            k === nothing && continue
            se = sqrt(mine["Var"][k] / mine["Nobs"][k] + pub["Var"][k] / pub["Nobs"][k])
            println(rpad(t, 8), rpad(@sprintf("%.4e", mine[column][k]), 14), rpad(@sprintf("%.4e", pub[column][k]), 14),
                se > 0 ? @sprintf("%+.1f", (mine[column][k] - pub[column][k]) / se) : "-")
        end
        println("rms of the difference / s.e. over all times: ", @sprintf("%.2f", sqrt(mean(abs2, z))), " (about 1 if both estimate the same function)")
    end
end

# ---------------------------------------------------------------------------------------------

function main(args)
    command, o = parse_options(args)
    commands = Dict("noise" => o -> run_statistic(o, "noise"), "kernel" => o -> run_statistic(o, "kernel"),
        "assemble" => assemble, "check" => check)
    haskey(commands, command) || error("usage: simulate_kernels.jl {$(join(keys(commands), "|"))} [options]; see the header of this file")
    commands[command](o)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main(ARGS)
end
