using DrWatson
import JSON
using Random
using FileIO
using ArgParse
using Dates

# my_args = Dict(
#     "script" => "simple_network.jl",
#     "system" => "JumpSystem",
#     # "scale" => 0.1,
#     "algorithm" => "annealing",
#     "run_name" => "2020-11-27",
#     "duration" => 2 .^ range(log2(0.1), log2(2.0), length=6),
#     "num_responses" => 50_000,
#     "mean_s" => [20, 40],
#     "corr_time_s" => 1,
#     "corr_time_ratio" => [2, 5, 10],
# )

my_args = Dict(
    "script" => "chemotaxis.jl",
    "run_name" => "2021-01-18",
    "num_responses" => 10_000,
    "duration" => 2,
    "mean_L" => 50,
    "num_receptors" => 10,
    "Y_tot" => 50,
    "LR_timescale" => 0.5,
    "Y_timescale" => 0.25
)

const NODES = 3
const PPN = 36
const QUEUE = "highcore"
const NAME = "CHEMOTAXIS"

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

function submit_job(out_dir, filename, runtime; job_before = nothing, dry_run=false)
    jobscript = """
        export JULIA_PROJECT=$(projectdir())

        # DEPOT=/dev/shm/julia_depot
        # mkdir -p $DEPOT
        # rsync -au ~/.julia/* $DEPOT
        # export JULIA_DEPOT_PATH=$DEPOT:~/.julia

        julia -e "using InteractiveUtils; versioninfo(verbose=true)"
        julia $(projectdir("scripts", "run_cluster.jl")) $(filename) $(my_args["script"])
        """

    name = NAME
    resources = `-l nodes=$NODES:ppn=$PPN:$QUEUE,mem=$(NODES * PPN * 4)gb,walltime=$runtime`

    if job_before !== nothing
        dependency = `-W depend=afterok:$job_before`
    else
        dependency = ``
    end

    cmd = `qsub -q $QUEUE -N $name $resources $dependency -j oe -o $out_dir`

    if dry_run
        println(cmd)
        return "$(rand(1:1000)).head.hollandia.amolf.nl"
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
    if dict["script"] == "chemotaxis.jl"
        return 24 * 60 * 60
    end

    if dict["algorithm"] == "annealing"
        factor = 0.14 * 1.5 # empirical factor from AMOLF cluster. The 1.5 is to make sure adequate headroom
    elseif dict["algorithm"] == "thermodynamic_integration"
        factor = 0.05 * 2.0 # empirical factor from AMOLF cluster. The 2.0 is to make sure adequate headroom
    elseif dict["algorithm"] == "directmc"
        factor = 0.03 * 2.0 # empirical factor from AMOLF cluster. The 2.0 is to make sure adequate headroom
    else
        error("unknown algorithm $(dict["algorithm"])")
    end
    constant = 20 * 60 + (NODES * PPN) * 5 # just make sure we have an extra buffer

    num_reactions = dict["duration"] * dict["mean_s"] * (1 / dict["corr_time_s"]) * (1 + dict["corr_time_ratio"])

    val = factor * (num_reactions + 10) * dict["num_responses"]
    round(Int, val / (NODES * PPN) + constant)
end

function submit_sims(; job_before=nothing, dry_run=false)
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
        job_before = submit_job(out_dir, f, runtime, job_before=job_before, dry_run=dry_run)
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
