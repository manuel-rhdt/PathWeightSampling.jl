module PathWeightSampling

include("marginal_strategies/strategies.jl")
include("trajectories/ssa.jl")
include("trajectories/path_mcmc.jl")
include("JumpSystem.jl")
include("ContinuousSystem.jl")
include("EmpiricalDistribution.jl")
include("write_hdf5.jl")
include("write_json.jl")
include("write_parquet.jl")
include("example_systems.jl")
include("ChemotaxisSMC.jl")
include("ParallelRun.jl")

using .AIS
using .DirectMC
using .SMC
using .ThermodynamicIntegration
using .FlatPerm
using .ChemotaxisSMC

using .JumpSystem
using .ContinuousSystem

export TIEstimate, AnnealingEstimate, DirectMCEstimate, SMCEstimate, AggregatedSMCEstimate, PERM,
    generate_configuration, mutual_information

end # module
