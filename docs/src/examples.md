
# Examples

## Simple Parallel Simulation

The script below reads in a JSON file as its first argument that sets up the simulation parameters. The input file could for example
look as follows:

```json
{   
    "params": {
        "duration": 10.0,
        "run_name": "test",
        "algorithm": "directmc",
        "num_samples": 500,
        "directmc_samples": 500,
        "mean_s": 50,
        "corr_time_s": 1,
        "corr_time_ratio": 10
    }
}
```

Then the example script reads this input file, sets up the "gene-expression" system with the correct parameters, and performs a parallel PWS simulation.
The results are stored in a HDF5 file, which is the input filename with the extension `.hdf5` appended.

```julia
using Distributed

@everywhere begin
    import JSON
    using Logging
    using HDF5
    using PWS
end

f = ARGS[1]
dict = JSON.parsefile(f)["params"]
@info "Read file" file = f

if !haskey(dict, "dtimes")
    dict["dtimes"] = collect(0.0:0.1:dict["duration"])
end

if dict["algorithm"] == "ti"
    algorithm = TIEstimate(0, 16, dict["ti_samples"])
elseif dict["algorithm"] == "annealing"
    algorithm = AnnealingEstimate(5, 100, 100)
elseif dict["algorithm"] == "directmc"
    algorithm = DirectMCEstimate(dict["directmc_samples"])
elseif dict["algorithm"] == "smc"
    algorithm = SMCEstimate(dict["smc_samples"])
else
    error("Unsupported algorithm " * dict["algorithm"])
end

@info "Parameters" dict
@info "Algorithm" algorithm

system_fn = () -> PWS.gene_expression_system(
    mean_s = dict["mean_s"],
    corr_time_s = dict["corr_time_s"],
    corr_time_x = dict["corr_time_s"] / dict["corr_time_ratio"],
    dtimes = dict["dtimes"]
)

mi = PWS.run_parallel(system_fn, algorithm, dict["num_samples"])
result = Dict(
    "Samples" => mi, 
    "Parameters" => dict
)

filename = f * ".hdf5"
h5open(filename, "w") do file
    PWS.write_hdf5!(file, result)
end
@info "Saved to" filename
```


