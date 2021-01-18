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

using HDF5
using Logging

@info "Loading GaussianMcmc"
using GaussianMcmc

if dict["algorithm"] == "thermodynamic_integration"
    algorithm = TIEstimate(1024, 6, 2^15)
elseif dict["algorithm"] == "annealing"
    algorithm = AnnealingEstimate(5, 100, 100)
elseif dict["algorithm"] == "directmc"
    algorithm = DirectMCEstimate(2^16)
else
    error("Unsupported algorithm " * dict["algorithm"])
end

@info "Parameters" run_name duration num_responses algorithm mean_s corr_time_s corr_time_ratio

import Catalyst: @reaction_network

extra_kwargs = Dict{Symbol,Any}()
if dict["system"] == "JumpSystem"
    sn = @reaction_network begin
        κ, ∅ --> S
        λ, S --> ∅
    end κ λ

    rn = @reaction_network begin
        ρ, S --> X + S
        μ, X --> ∅ 
    end ρ μ

    @everywhere workers() system = JumpSystem($sn, $rn, [$κ, $λ], [$ρ, $μ], s0=$mean_s, x0=$mean_x, duration=$duration)
elseif dict["system"] == "GaussianSystem"
    extra_kwargs[:scale] = 0.08
    @everywhere workers() system = GaussianSystem($κ, $λ, $ρ, $μ, delta_t=0.05, duration=$duration)
else
    error("unknown system $(dict["system"])")
end


using Distributions
using LinearAlgebra

function reduce_results(res1, results...)
    new_res = typeof(res1)()
    for k in keys(res1)
        new_res[k] = vcat(res1[k], (r[k] for r in results)...)
    end
    new_res
end

@info "Generated initial configuration"

me = pmap(x -> marginal_entropy(system, algorithm=algorithm; num_responses=20, extra_kwargs...), 1:div(num_responses, 20, RoundUp))
me = reduce_results(me...)

@info "Finished marginal entropy"

ce = pmap(x -> conditional_entropy(system, num_responses=1000), 1:num_responses)
ce = reduce_results(ce...)

@info "Finished conditional entropy"

function DrWatson._wsave(filename, result::Dict)
    h5open(filename, "w") do file
        GaussianMcmc.write_hdf5!(file, result)
    end
end


filename = savename((@dict duration mean_s corr_time_ratio), "hdf5")
local_path = datadir(dict["algorithm"], run_name, filename)
tagsave(local_path, merge(dict, me, ce), storepatch=false)
@info "Saved to" filename

# upload to SUN storage

include("smbclient.jl")
sun_path = joinpath(sun_home, "data", dict["algorithm"], run_name, filename)
mkpath(dirname(sun_path))
cp(local_path, sun_path)
