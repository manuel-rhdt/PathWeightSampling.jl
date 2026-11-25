#!/usr/bin/env julia
#
# Download the raw simulation data (HDF5 files) behind the chemotaxis figures from Zenodo and verify them.
#
#     julia figures/figure_rates/download_data.jl                 # all runs
#     julia figures/figure_rates/download_data.jl 2023-02-17      # one run (the 400-cluster, M = 256 run)
#
# The files are written to `data/raw/<run>/` (relative to the repository root) and checked against
# `MANIFEST.sha256` of the record. Uses only the Julia standard library.
# See PROVENANCE.md for which run belongs to which figure file.

using Downloads, SHA, Tar

# Zenodo record of the data, https://doi.org/10.5281/zenodo.24530700
const RECORD = get(ENV, "PWS_DATA_RECORD", "24530700")
const RUNS = ["2023-02-17", "2023-01-15", "2023-01-11"]
const OUT = get(ENV, "PWS_DATA_OUT", joinpath(@__DIR__, "..", "..", "data", "raw"))

# PWS_DATA_BASE_URL overrides the location (for mirrors and tests)
function file_url(name)
    base = get(ENV, "PWS_DATA_BASE_URL", "https://zenodo.org/records/$RECORD/files")
    "$base/$name" * (haskey(ENV, "PWS_DATA_BASE_URL") ? "" : "?download=1")
end

function fetch_file(name, dest)
    isfile(dest) && return dest
    mkpath(dirname(dest))
    println("downloading ", name)
    Downloads.download(file_url(name), dest)
    dest
end

function read_manifest(path)
    entries = Dict{String,String}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        hash, file = split(line, "  "; limit=2)
        entries[file] = hash
    end
    entries
end

function main(args)
    runs = isempty(args) ? RUNS : args
    all(in(RUNS), runs) || error("unknown run in $runs; available: $RUNS")
    mkpath(OUT)
    manifest = read_manifest(fetch_file("MANIFEST.sha256", joinpath(OUT, "MANIFEST.sha256")))
    bad = String[]
    for run in runs
        tarfile = fetch_file("runs_$run.tar", joinpath(OUT, "runs_$run.tar"))
        dest = joinpath(OUT, run)
        isdir(dest) || Tar.extract(tarfile, dest)
        files = [f for f in keys(manifest) if startswith(f, run * "/")]
        for f in sort(files)
            path = joinpath(OUT, f)
            if !isfile(path) || bytes2hex(open(sha256, path)) != manifest[f]
                push!(bad, f)
            end
        end
        println(run, ": ", length(files), " files checked")
    end
    isempty(bad) || error("checksum mismatch or missing: ", join(bad, ", "))
    println("all files match MANIFEST.sha256")
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
