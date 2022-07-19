module PathWeightSampling

using Distributions
import DiffEqBase
using JumpProcesses
include("trajectories/trajectory.jl")
include("EventIter.jl")
include("MetropolisSampler.jl")
include("marginal_strategies/strategies.jl")
include("GaussianSystem.jl")
include("trajectories/distribution.jl")
include("EmpiricalDistribution.jl")
include("model_setup/DrivenJumpProblem.jl")
include("JumpNetwork.jl")
include("write_hdf5.jl")

include("example_systems.jl")

include("ParallelRun.jl")

using .DrivenJumpProblems

export TIEstimate, AnnealingEstimate, DirectMCEstimate, SMCEstimate,
    generate_configuration, mutual_information

end # module
