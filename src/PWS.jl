module PWS

using Distributions
using DiffEqBase: AbstractJumpProblem
using DiffEqJump: DiffEqBase
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

export JumpSystem, 
    GaussianSystem, marginal_entropy, conditional_entropy, 
    TIEstimate, AnnealingEstimate, DirectMCEstimate, SMCEstimate,
    generate_configuration, SRXsystem,
    mutual_information

end # module
