struct DirectMCEstimate
    num_samples::Int
end

name(x::DirectMCEstimate) = "Direct MC"

struct DirectMCResult
    samples::Vector{Float64}
end

function summary(results::DirectMCResult...)
    block_size = 2^14
    logmeanexp(x) = log(mean(exp.(x .- maximum(x)))) + maximum(x)
    blocks = [logmeanexp.(Iterators.partition(est.samples, block_size)) for est in results]
    DataFrame(Blocks=blocks)
end

log_marginal(result::DirectMCResult) = -(logsumexp(result.samples) - log(length(result.samples)))
function Statistics.var(result::DirectMCResult)
    max_weight = maximum(result.samples)
    log_mean_weight = max_weight + log(mean(exp.(result.samples .- max_weight)))
    log_var = log(var(exp.(result.samples .- max_weight))) + 2 * max_weight
    log_var -= log(length(result.samples))

    exp(-2log_mean_weight + log_var)
end


function simulate(algorithm::DirectMCEstimate, initial, system; kwargs...)
    samples = zeros(Float64, algorithm.num_samples)
    for i in 1:algorithm.num_samples
        signal = new_signal(initial, system)
        samples[i] = -energy(signal, system, 1.0)
    end
    DirectMCResult(samples)
end