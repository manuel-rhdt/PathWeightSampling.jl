module PathWeightSampling

using Distributions
import DiffEqBase

include("MetropolisSampler.jl")
include("marginal_strategies/strategies.jl")
include("GaussianSystem.jl")
include("trajectories/ssa.jl")
include("JumpSystem.jl")
include("EmpiricalDistribution.jl")
include("write_hdf5.jl")

include("example_systems.jl")

include("ParallelRun.jl")

export TIEstimate, AnnealingEstimate, DirectMCEstimate, SMCEstimate,
    generate_configuration, mutual_information

end # module
