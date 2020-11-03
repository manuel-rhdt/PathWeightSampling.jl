using DrWatson
import JSON
using Random
using FileIO

my_args = Dict(
    "algorithm" => "thermodynamic_integration",
    "run_name" => "2020-11-03_stationary_sweep",
    "duration" => 2 .^ range(log2(20), log2(200), length=6),
    "num_responses" => 1000,
    "mean_s" => [20, 40, 70, 100],
    "corr_time_s" => 100,
    "corr_time_ratio" => 5,
)

function runsave(dicts, tmp=projectdir("_research", "tmp"), prefix="", suffix="json", l=8)
    mkpath(tmp)
    n = length(dicts)
    indices = map(string, 1:n)
    existing = readdir(tmp)
    filename = prefix * randstring(l)
    r = filename .* "." .* indices .* "." .* suffix
    while !isdisjoint(existing, r)
        filename = prefix * randstring(l)
        r = filename .* "." .* indices .* "." .* suffix
    end

    for (i, path) ∈ enumerate(r)
        open(joinpath(tmp, path), "w") do io
            JSON.print(io, copy(dicts[i]))
        end
    end

    filename
end

dicts = dict_list(my_args)

function DrWatson._wsave(filename, d::Dict)
    if endswith(filename, ".json")
        open(filename, "w") do io
            JSON.print(io, d)
        end
    else
        FileIO.save(filename, d)
    end
end

filenames = tmpsave(dicts, suffix="json")

out_dir = projectdir("data", "output")
mkpath(out_dir)

function submit_job_array(filename, njobs, runtime)
    jobscript = """
        export JULIA_PROJECT=$(projectdir())

        julia -e "using InteractiveUtils; versioninfo(verbose=true)"
        julia -O3 $(projectdir("scripts", "simple_network.jl")) $filename.json
    """
    
    result = ""
    open(`qsub -N Sweep_S -l nodes=1:ppn=1:highcore,mem=4gb,walltime=$runtime -t 1-$njobs -j oe -o $out_dir`, "r+") do io
        print(io, jobscript)
        close(io.in)
        result *= read(io, String)
    end
    
    println(result)
end

function estimate_runtime(dict)
    factor = 0.0015 * 1.8
    constant = 5 * 60
    round(Int, factor * dict["mean_s"] * dict["duration"] * dict["num_responses"] + constant)
end

for d in dicts
    runtime = estimate_runtime(d)
    submit_job_array(d, 100, runtime)
end
