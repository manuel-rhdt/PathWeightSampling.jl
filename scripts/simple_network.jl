@everywhere begin
    using DrWatson
    import JSON
    using Logging
    using HDF5
    using GaussianMcmc
end

f = ARGS[1]
dict = JSON.parsefile(projectdir("_research", "tmp", f))
@info "Read file" file = projectdir("_research", "tmp", f)

duration = dict["duration"]

num_responses = dict["num_responses"]
run_name = dict["run_name"]

mean_s = dict["mean_s"]
corr_time_s = dict["corr_time_s"]
corr_time_ratio = dict["corr_time_ratio"]
dtimes = collect(0.0:0.04:duration)

if dict["algorithm"] == "thermodynamic_integration"
    algorithm = TIEstimate(1024, 6, 2^15)
elseif dict["algorithm"] == "annealing"
    algorithm = AnnealingEstimate(5, 100, 100)
elseif dict["algorithm"] == "directmc"
    algorithm = DirectMCEstimate(2^16)
elseif dict["algorithm"] == "smc"
    algorithm = SMCEstimate(dict["smc_samples"])
else
    error("Unsupported algorithm " * dict["algorithm"])
end

@info "Parameters" run_name duration num_responses algorithm mean_s corr_time_s corr_time_ratio

system = GaussianMcmc.gene_expression_system(
    mean_s=mean_s,
    corr_time_s = corr_time_s,
    corr_time_x = corr_time_s / corr_time_ratio,
    dtimes=dtimes
)

mi = GaussianMcmc.run_parallel(system, algorithm, num_responses)
result = Dict("Samples" => mi, "DiscreteTimes" => dtimes)

function DrWatson._wsave(filename, result::Dict)
    h5open(filename, "w") do file
        GaussianMcmc.write_hdf5!(file, result)
    end
end

save_dict = Dict("Alg" => dict["algorithm"], "Duration" => duration, "Smean" => mean_s, "Xtimescale" => corr_time_s / corr_time_ratio)
filename = savename(save_dict, "hdf5")
local_path = datadir("gene_expression", run_name, filename)
tagsave(local_path, merge(dict, result), storepatch=false)
@info "Saved to" filename

# upload to SUN storage

include("smbclient.jl")
sun_path = joinpath(sun_home, "data", dict["algorithm"], run_name, filename)
mkpath(dirname(sun_path))
cp(local_path, sun_path)
