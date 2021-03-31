module GaussianMcmc

include("trajectories/trajectory.jl")
include("EventIter.jl")
include("MetropolisSampler.jl")
include("marginal_strategies/strategies.jl")
include("GaussianSystem.jl")
include("trajectories/distribution.jl")
include("JumpNetwork.jl")
include("write_hdf5.jl")

include("example_systems.jl")

include("ParallelRun.jl")

export JumpSystem, 
    GaussianSystem, marginal_entropy, conditional_entropy, 
    TIEstimate, AnnealingEstimate, DirectMCEstimate, SMCEstimate,
    generate_configuration, SRXsystem,
    mutual_information

end # module
