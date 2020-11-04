using DrWatson
import JSON
using Random
using FileIO

my_args = Dict(
    "algorithm" => "annealing",
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
        julia -O3 $(projectdir("scripts", "simple_network.jl")) $(filename)
        """
    
    result = ""
    open(`qsub -N Annealing -l nodes=1:ppn=1:highcore,mem=4gb,walltime=$runtime -t 1-$njobs -j oe -o $out_dir`, "r+") do io
        print(io, jobscript)
        close(io.in)
        result *= read(io, String)
    end
    
    print(result)
end

function estimate_runtime(dict)
    if dict["algorithm"] == "annealing"
        factor = 0.0005 * 1.5 # empirical factor from AMOLF cluster. The 1.5 is to make sure adequate headroom
    elseif dict["algorithm"] == "thermodynamic_integration"
        factor = 0.002 * 1.5 # empirical factor from AMOLF cluster. The 1.5 is to make sure adequate headroom
    else
        error("unknown algorithm $(dict["algorithm"])")
    end
    constant = 20 * 60 # just make sure we have an extra buffer of 20 minutes
    round(Int, factor * dict["mean_s"] * dict["duration"] * dict["num_responses"] + constant)
end

for (d, f) in zip(dicts, filenames)
    runtime = estimate_runtime(d)
    submit_job_array(f, 100, runtime)
end
