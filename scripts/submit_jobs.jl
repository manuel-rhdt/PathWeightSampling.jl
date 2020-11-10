using DrWatson
import JSON
using Random
using FileIO
using ArgParse
using Dates

my_args = Dict(
    "algorithm" => "thermodynamic_integration",
    "run_name" => "2020-11-10_S=100",
    "duration" => 2 .^ range(log2(20), log2(200), length=5),
    "num_responses" => 1200,
    "mean_s" => 100,
    "corr_time_s" => 100,
    "corr_time_ratio" => 5,
)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dry-run"
            help = "only show qsub commands"
            action = :store_true
        "--time"
            help = "only show estimated runtime"
            action = :store_true
    end
    parse_args(ARGS, s)
end

function DrWatson._wsave(filename, d::Dict)
    if endswith(filename, ".json")
        open(filename, "w") do io
            JSON.print(io, d)
        end
    else
        FileIO.save(filename, d)
    end
end

function submit_job_array(out_dir, filename, njobs, runtime; array_before = nothing, dry_run=false)
    jobscript = """
        export JULIA_PROJECT=$(projectdir())

        julia -e "using InteractiveUtils; versioninfo(verbose=true)"
        julia -O3 $(projectdir("scripts", "simple_network.jl")) $(filename)
        """

    name = "TI_NOV_10"
    resources = `-l nodes=1:ppn=1:highcore,mem=4gb,walltime=$runtime`

    if array_before !== nothing
        dependency = `-W depend=afterokarray:$array_before`
    else
        dependency = ``
    end

    cmd = `qsub -N $name $resources $dependency -t 1-$njobs -j oe -o $out_dir`

    if dry_run
        println(cmd)
        return "$(rand(1:1000))[].head.hollandia.amolf.nl"
    end

    result = ""
    open(cmd, "r+") do io
        print(io, jobscript)
        close(io.in)
        result *= read(io, String)
    end
    
    print(result)
    strip(result)
end

function estimate_runtime(dict)
    if dict["algorithm"] == "annealing"
        factor = 0.0007 * 1.5 # empirical factor from AMOLF cluster. The 1.5 is to make sure adequate headroom
    elseif dict["algorithm"] == "thermodynamic_integration"
        factor = 0.002 * 1.5 # empirical factor from AMOLF cluster. The 1.5 is to make sure adequate headroom
    else
        error("unknown algorithm $(dict["algorithm"])")
    end
    constant = 20 * 60 # just make sure we have an extra buffer of 20 minutes
    round(Int, factor * dict["mean_s"] * dict["duration"] * dict["num_responses"] + constant)
end

function submit_sims(; array_before=nothing, dry_run=false)
    dicts = dict_list(my_args)

    out_dir = projectdir("data", "output")
    if !dry_run
        filenames = tmpsave(dicts, suffix="json")
        mkpath(out_dir)
    else
        filenames = [randstring() for i=1:length(dicts)]
    end

    for (d, f) in zip(dicts, filenames)
        runtime = estimate_runtime(d)
        array_before = submit_job_array(out_dir, f, 36*6, runtime, array_before=array_before, dry_run=dry_run)
    end
end

function print_times()
    sum = Dates.CompoundPeriod()
    dicts = dict_list(my_args)
    for (i, d) in enumerate(dicts)
        runtime = Dates.CompoundPeriod(Second(estimate_runtime(d)))
        println("job $i: $(canonicalize(runtime))")
        sum += runtime
    end

    println("sum: $(canonicalize(sum))")
end

function main()
    parsed_args = parse_commandline()

    if parsed_args["time"]
        print_times()
    else
        submit_sims(dry_run = parsed_args["dry-run"])
    end
end

main()
