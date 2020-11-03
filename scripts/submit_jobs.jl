using DrWatson
import JSON
using Random

my_args = Dict(
    "algorithm" => "thermodynamic_integration",
    "run_name" => "2020-11-03_stationary_sweep",
    "duration" => 2 .^ range(log2(20), log2(500), length=6),
    "num_responses" => 1000,
    "mean_s" => [20, 50, 100, 200],
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

out_dir = projectdir("data", "output")
mkpath(out_dir)

function submit_job_array(dict, njobs)
    jobarray = collect(1:njobs)

    dict = copy(dict)

    dict["N"] = jobarray
    array_dicts = dict_list(dict)

    filename = runsave(dicts)
    jobscript = """
        export JULIA_PROJECT=$(projectdir())

        julia -e "using InteractiveUtils; versioninfo(verbose=true)"
        julia -O3 $(projectdir("scripts", "simple_network.jl")) $filename.\$PBS_ARRAYID.json
    """
    
    result = ""
    open(`qsub -N Sweep_S -l nodes=1:ppn=1:highcore,mem=4gb,walltime=10:00:00 -t 1-$njobs -j oe -o $out_dir`, "r+") do io
        print(io, jobscript)
        close(io.in)
        global result *= read(io, String)
    end
    
    print(result)
end

for d in dicts
    submit_job_array(d, 100)
end
