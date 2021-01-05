using DrWatson
import JSON
using Logging

f = ARGS[1]
dict = JSON.parsefile(projectdir("_research", "tmp", f))
@info "Read file" file = projectdir("_research", "tmp", f)

run_name = dict["run_name"]
duration = dict["duration"]
num_responses = dict["num_responses"]

mean_L = dict["mean_L"]
num_receptors = dict["num_receptors"]
Y_tot = dict["Y_tot"]
LR_ratio = 0.5
Y_ratio = 0.5

LR_timescale = dict["LR_timescale"]
Y_timescale = dict["Y_timescale"]

mean_LR = num_receptors * LR_ratio
mean_R = num_receptors - mean_LR

mean_Yp = Y_tot * Y_ratio
mean_Y = Y_tot - mean_Yp

using HDF5
using Distributed
import Catalyst: @reaction_network

@info "Loading GaussianMcmc"
@everywhere using GaussianMcmc
@info "Done"

using StaticArrays

extra_kwargs = Dict{Symbol,Any}()

sn = @reaction_network begin
    κ, ∅ --> L
    λ, L --> ∅
end κ λ

rn = @reaction_network begin
    ρ, L + R --> L + LR
    μ, LR --> R
end ρ μ

xn = @reaction_network begin
    δ, LR + Y --> Yp + LR
    χ, Yp --> Y
end δ χ

u0 = SA[mean_L, num_receptors, 0, Y_tot, 0]
tspan = (0.0, duration)
ps = [mean_L, 1.0]
pr = [mean_LR / (LR_timescale * mean_R * mean_L), 1 / LR_timescale]
px = [mean_Yp / (Y_timescale * mean_Y * mean_LR), 1 / Y_timescale]



system = SRXsystem(sn, rn, xn, u0, ps, pr, px, tspan)
algorithm = DirectMCEstimate(50_000)

batch_size = 10
result = pmap(x -> mutual_information(system, algorithm; num_responses=batch_size, extra_kwargs...), 1:div(num_responses, batch_size, RoundUp))

function reduce_results(res1, results...)
    new_res = typeof(res1)()
    for k in keys(res1)
        new_res[k] = vcat(res1[k], (r[k] for r in results)...)
    end
    new_res
end

result = reduce_results(result...)

function DrWatson._wsave(filename, result::Dict)
    h5open(filename, "w") do file
        GaussianMcmc.write_hdf5!(file, result)
    end
end

filename = savename((@dict duration), "hdf5")
local_path = datadir("chemotaxis", run_name, filename)
tagsave(local_path, merge(dict, result), storepatch=false)
@info "Saved to" filename

# upload to SUN storage

include("smbclient.jl")
sun_path = joinpath(sun_home, "data", "chemotaxis", run_name, filename)
mkpath(dirname(sun_path))
cp(local_path, sun_path)
