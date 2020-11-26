module GaussianMcmc

include("MetropolisSampler.jl")
include("GaussianSystem.jl")
include("JumpSystem.jl")
include("marginal_strategies/strategies.jl")
include("write_hdf5.jl")

export JumpSystem, 
    GaussianSystem, marginal_entropy, conditional_entropy, 
    TIEstimate, AnnealingEstimate, DirectMCEstimate,
    generate_configuration

end # module
