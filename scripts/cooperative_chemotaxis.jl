@everywhere begin
    using DrWatson
    import JSON
    using Logging
    using HDF5
    using GaussianMcmc
end

f = ARGS[1]
dict = JSON.parsefile(projectdir("_research", "tmp", f))["params"]
@info "Read file" file = projectdir("_research", "tmp", f)

run_name = dict["run_name"]
duration = dict["duration"]
num_responses = dict["num_responses"]

# mean_L = dict["mean_L"]
# num_receptors = dict["num_receptors"]
# Y_tot = dict["Y_tot"]

params = (;
	E₀ = 0.0,
	lmax = 3,
	mmax = 9,
	Kₐ = 500,
	Kᵢ = 25,
	δf = -1.0,
	n_clusters = 800,
	a_star = 0.8,
	k⁺ = 0.05,
	n_chey = 10_000,
	mean_l = 50,
)

save_dict = Dict("Alg" => dict["algorithm"], "Duration" => duration, "M" => dict["smc_samples"])

dtimes = collect(0.0:0.5:duration)

system_fn = () -> GaussianMcmc.cooperative_chemotaxis_system(dtimes = dtimes; params...)

algorithm = SMCEstimate(dict["smc_samples"])

mi = GaussianMcmc.run_parallel(system_fn, algorithm, num_responses)
result = Dict("Samples" => mi, "DiscreteTimes" => dtimes)

function DrWatson._wsave(filename, result::Dict)
    h5open(filename, "w") do file
        GaussianMcmc.write_hdf5!(file, result)
    end
end

filename = savename(save_dict, "hdf5")
local_path = datadir("coop_chemotaxis", run_name, filename)
tagsave(local_path, merge(dict, result), storepatch=false)
@info "Saved to" filename

# upload to SUN storage

include("smbclient.jl")
sun_path = joinpath(sun_home, "data", "coop_chemotaxis", run_name, filename)
mkpath(dirname(sun_path))
cp(local_path, sun_path)
