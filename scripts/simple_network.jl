using DrWatson
import JSON
using Distributed

f = ARGS[1]
dict = JSON.parsefile(projectdir("_research", "tmp", f))
@info "Read file" file = projectdir("_research", "tmp", f)

duration = dict["duration"]

num_responses = dict["num_responses"]
run_name = dict["run_name"]

mean_s = dict["mean_s"]
corr_time_s = dict["corr_time_s"]
corr_time_ratio = dict["corr_time_ratio"]

λ = 1 / corr_time_s
κ = mean_s * λ
μ = corr_time_ratio / corr_time_s
ρ = μ
mean_x = mean_s

@everywhere workers() using GaussianMcmc.Trajectories
using HDF5
using Logging
using Catalyst

if dict["algorithm"] == "thermodynamic_integration"
    algorithm = TIEstimate(1024, 6, 2^15)
elseif dict["algorithm"] == "annealing"
    algorithm = AnnealingEstimate(15, 50, 100)
elseif dict["algorithm"] == "directmc"
    algorithm = DirectMCEstimate(2^16)
else
    error("Unsupported algorithm " * dict["algorithm"])
end

@info "Parameters" run_name duration num_responses algorithm mean_s corr_time_s corr_time_ratio

sn = @reaction_network begin
    κ, ∅ --> S
    λ, S --> ∅
end κ λ

rn = @reaction_network begin
    ρ, S --> X + S
    μ, X --> ∅ 
end ρ μ

using Distributions
using LinearAlgebra

function reduce_results(res1, results...)
    new_res = typeof(res1)()
    for k in keys(res1)
        new_res[k] = vcat(res1[k], (r[k] for r in results)...)
    end
    new_res
end

@everywhere workers() gen = Trajectories.configuration_generator($sn, $rn, [$κ, $λ], [$ρ, $μ], $mean_s, $mean_x)

@info "Generated initial configuration"

marginal_entropy = pmap(x -> Trajectories.marginal_entropy(gen, algorithm=algorithm; num_responses=20, duration=duration), 1:div(num_responses, 20, RoundUp))
marginal_entropy = reduce_results(marginal_entropy...)

@info "Finished marginal entropy"

conditional_entropy = pmap(x -> Trajectories.conditional_entropy(gen, num_responses=200, duration=duration), 1:num_responses)
conditional_entropy = reduce_results(conditional_entropy...)

@info "Finished conditional entropy"

function DrWatson._wsave(filename, result::Dict)
    h5open(filename, "w") do file
        Trajectories.write_hdf5!(file, result)
    end
end


filename = savename((@dict duration mean_s corr_time_ratio), "hdf5")
local_path = datadir(dict["algorithm"], run_name, filename)
tagsave(local_path, merge(dict, marginal_entropy, conditional_entropy), storepatch=false)
@info "Saved to" filename

# upload to SUN storage

include("smbclient.jl")
sun_path = joinpath(sun_home, "data", dict["algorithm"], run_name, filename)
mkpath(dirname(sun_path))
cp(local_path, sun_path)
