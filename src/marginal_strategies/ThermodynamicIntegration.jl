import Statistics
using FastGaussQuadrature

struct TIEstimate
    burn_in::Int
    integration_nodes::Int
    num_samples::Int
end

name(x::TIEstimate) = "TI"

struct ThermodynamicIntegrationResult <: SimulationResult
    integration_weights::Vector{Float64}
    inv_temps::Vector{Float64}
    energies::Array{Float64,2}
    acceptance::Array{Bool,2}
end

function blocks(result::ThermodynamicIntegrationResult, block_size=2^10)
    block_averages(array) = map(mean, Iterators.partition(array, block_size))
    mapreduce(block_averages, hcat, eachcol(result.energies))
end

# perform the quadrature integral
log_marginal(result::ThermodynamicIntegrationResult) = dot(result.integration_weights, vec(mean(result.energies, dims=1)))
function Statistics.var(result::ThermodynamicIntegrationResult, block_size=2^10)
    b = blocks(result, block_size)
    σ² = var(b, dims=1) ./ size(b, 1)
    dot(result.integration_weights.^2, σ²)
end

function summary(results::ThermodynamicIntegrationResult...)
    block_size = 2^10
    energy_blocks = [blocks(r, block_size) for r in results]
    acceptance = [mean(r.acceptance, dims=1) for r in results]
    inv_temps = [r.inv_temps for r in results]
    integration_weights = [r.integration_weights for r in results]

    DataFrame(EnergyBlocks=energy_blocks, Acceptance=acceptance, Theta=inv_temps, IntegrationWeights=integration_weights)
end


# Monte-Carlo computation of the marginal probability for the given configuration
function simulate(algorithm::TIEstimate, initial, system; kwargs...)
    # Generate the array of θ values for which we want to simulate the system.
    # We use Gauss-Legendre quadrature which predetermines the choice of θ.
    nodes, weights = gausslegendre(algorithm.integration_nodes)
    θrange = 0.5 .* nodes .+ 0.5
    # The factor 0.5 comes from rescaling the integration limits from [-1,1] to [0,1].
    weights = 0.5 .* weights

    energies = zeros(Float64, algorithm.num_samples, length(θrange))
    accept = Array{Bool}(undef, algorithm.num_samples, length(θrange))
    for i in eachindex(θrange)
        chn = chain(system; θ=θrange[i], kwargs...)
        sampler = MetropolisSampler(algorithm.burn_in, 0, energy(initial, chn), copy(initial), chn)
        for (j, was_accepted) in Iterators.enumerate(Iterators.take(sampler, algorithm.num_samples))
            energies[j, i] = energy_difference(sampler.state, system)
            accept[j, i] = was_accepted != 0
        end
    end

    ThermodynamicIntegrationResult(weights, θrange, energies, accept)
end
