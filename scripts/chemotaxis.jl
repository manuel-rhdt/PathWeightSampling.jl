using DrWatson
import JSON
using Distributed

f = ARGS[1]
dict = JSON.parsefile(projectdir("_research", "tmp", f))
@info "Read file" file = projectdir("_research", "tmp", f)

run_name = dict["run_name"]
duration = dict["duration"]

using HDF5
using Logging

import Catalyst: @reaction_network

@info "Loading GaussianMcmc"
@everywhere using GaussianMcmc
@info "Done"

using StaticArrays

sn = @reaction_network begin
    κ, ∅ --> 2L
    λ, L --> ∅
end κ λ

rn = @reaction_network begin
    ρ, L + R --> L + LR
    μ, LR --> R
    ξ, R + CheY --> R + CheYp
    ν, CheYp --> CheY
end ρ μ ξ ν

xn = @reaction_network begin
    δ, CheYp --> CheYp + X
    χ, X --> ∅
end δ χ

u0 = SA[10, 30, 0, 50, 0, 0]
tspan = (0.0, duration)
ps = [5.0, 1.0]
pr = [1.0, 4.0, 1.0, 2.0]
px = [1.0, 1.0]

system = SRXsystem(sn, rn, xn, u0, ps, pr, px, tspan)
algorithm = DirectMCEstimate(50_000)

result = pmap(x -> mutual_information(system, algorithm; num_responses=20, extra_kwargs...), 1:div(num_responses, 20, RoundUp))

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
tagsave(local_path, merge(dict, me, ce), storepatch=false)
@info "Saved to" filename

# upload to SUN storage

include("smbclient.jl")
sun_path = joinpath(sun_home, "data", "chemotaxis", run_name, filename)
mkpath(dirname(sun_path))
cp(local_path, sun_path)
