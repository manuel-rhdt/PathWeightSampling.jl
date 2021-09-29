using DrWatson
import JSON
using Random
using FileIO
using ArgParse
using Dates

# my_args = Dict(
#     "script" => "simple_network.jl",
#     "algorithm" => "directmc",
#     "directmc_samples" => 2 .^ [3, 4, 5, 6, 7, 8, 9, 10, 11],
#     "run_name" => "2021-06-17",
#     "duration" => 20,
#     "num_responses" => 10_000,
#     "mean_s" => [50],
#     "corr_time_s" => 1,
#     "corr_time_ratio" => 10,
# )

# my_args = Dict(
#     "script" => "simple_network.jl",
#     "algorithm" => "smc",
#     "smc_samples" => 2 .^ [3, 4, 5, 6, 7, 8, 9, 10, 11],
#     "run_name" => "2021-06-21",
#     "duration" => 20,
#     "num_responses" => 10_000,
#     "mean_s" => 50,
#     "corr_time_s" => 1,
#     "corr_time_ratio" => 10,
# )

# my_args = Dict(
#     "script" => "simple_network.jl",
#     "algorithm" => "ti",
#     "ti_samples" => 256,
#     "run_name" => "2021-06-18",
#     "duration" => [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0],
#     "num_responses" => 10_000,
#     "mean_s" => 50,
#     "corr_time_s" => 1,
#     "corr_time_ratio" => 10,
# )

# my_args = Dict(
#     "script" => "chemotaxis.jl",
#     "run_name" => "2021-04-05",
#     "smc_samples" => 64,
#     "num_responses" => 5_000,
#     "duration" => 20,
#     "mean_L" => 50,
#     "num_receptors" => 10_000,
#     "Y_tot" => 5000,
#     "LR_timescale" => 0.01,
#     "Y_timescale" => 0.1
# )

my_args = Dict(
    "script" => "cooperative_chemotaxis.jl",
    "algorithm" => "smc",
    "run_name" => "2021-09-29",
    "duration" => 200,
    "tau_l" => collect(range(0.1, 10.0, length=20)),
    "num_responses" => 3600,
    "smc_samples" => 128,
)

const NCPUS = 2 * 36
const QUEUE = "highcore"
const NAME = "COOP_CHEM"

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


        julia -e "using InteractiveUtils; versioninfo(verbose=true)"
        # julia -J$(projectdir())/GMcmcSysimage.so $(projectdir("scripts", "run_cluster.jl")) $(filename)  $(my_args["script"])
        julia $(projectdir("scripts", "run_cluster.jl")) $(filename)  $(my_args["script"])
        """

    name = NAME
    resources = `-l walltime=$runtime -l select=$NCPUS:ncpus=1:mem=4gb -l place=free`

    # if job_before !== nothing
    #     dependency = `-W depend=afterok:$job_before`
    # else
        dependency = ``
    # end

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
    if dict["script"] != "simple_network.jl"
        return 6 * 24 * 60 * 60
    end

    if dict["algorithm"] == "annealing"
        factor = 0.14 * 1.5 # empirical factor from AMOLF cluster. The 1.5 is to make sure adequate headroom
    elseif dict["algorithm"] == "ti"
        factor = 0.05 * 2.0 # empirical factor from AMOLF cluster. The 2.0 is to make sure adequate headroom
    elseif dict["algorithm"] == "directmc"
        factor = 0.03 * 2.0 # empirical factor from AMOLF cluster. The 2.0 is to make sure adequate headroom
    elseif dict["algorithm"] == "smc"
        factor = 0.1 * 2
    else
        error("unknown algorithm $(dict["algorithm"])")
    end
    constant = 20 * 60 + NCPUS * 5 # just make sure we have an extra buffer

    num_reactions = dict["duration"] * dict["mean_s"] * (1 / dict["corr_time_s"]) * (1 + dict["corr_time_ratio"])

    val = factor * (num_reactions + 10) * dict["num_responses"]
    round(Int, val / NCPUS + constant)
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
