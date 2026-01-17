using DataFrames
using ProgressMeter

import Random
import Statistics: mean

import Distributed: pmap

abstract type AbstractSystem end

abstract type SimulationResult end

discrete_times(::AbstractSystem) = error("Custom system does not implement required function `discrete_times`.")
generate_configuration(::AbstractSystem) = error("Custom system does not implement required function `generate_configuration`.")
conditional_density(::AbstractSystem, algorithm, configuration) = error("Custom system does not implement required function `conditional_density`.")
marginal_density(::AbstractSystem, algorithm, configuration) = error("Custom system does not implement required function `marginal_density`.")
compile(sys::AbstractSystem) = sys
to_dataframe(::Any) = DataFrame()

function information_density(s::AbstractSystem, algorithm, configuration; kwargs...)
    cond = conditional_density(s, algorithm, configuration; full_result=true, kwargs...)
    marg = marginal_density(s, algorithm, configuration; full_result=true, kwargs...)

    metrics = DataFrame()
    if cond isa SimulationResult
        df = to_dataframe(cond)
        df[!, :Conditional] .= true
        metrics = vcat(metrics, df, cols=:union)
        cond = log_marginal(cond)
    end
    if marg isa SimulationResult
        df = to_dataframe(marg)
        df[!, :Conditional] .= false
        metrics = vcat(metrics, df, cols=:union)
        marg = log_marginal(marg)
    end

    # ln [P(x,s)/(P(x)P(s))] = ln [P(x|s)/P(x)] = ln P(x|s) - ln P(x)
    replace(cond - marg, -Inf => missing, NaN => missing), metrics
end

log_marginal(::SimulationResult) = error("Custom subtype of SimulationResult does not implement required function `log_marginal`.")

abstract type AbstractSimulationAlgorithm end

simulate(s::AbstractSimulationAlgorithm, args...) = error("Unknown simulation algorihm", s)
name(x::AbstractSimulationAlgorithm) = string(typeof(x))

function _logmeanexp(x::AbstractArray)
    x_max = maximum(x)
    if x_max == -Inf
        return -Inf
    end
    log(mean(xi -> exp(xi - x_max), x)) + x_max
end

"""
    logmeanexp(x[; dims=nothing])

Compute log(mean(exp(x))) in a numerically stable way.
"""
logmeanexp(x::AbstractArray; dims=nothing) =
    if dims === nothing
        _logmeanexp(x)
    else
        mapslices(_logmeanexp, x, dims=dims)
    end

include("MetropolisSampler.jl")
include("ThermodynamicIntegration.jl")
include("AIS.jl")
include("SMC.jl")
include("DirectMC.jl")
include("flatPERM.jl")

"""
    mutual_information(system, algorithm; num_samples=1, progress=true)

Perform a simulation to compute the mutual information between input
and output trajectories of `system`. 

# Arguments

The required marginalization integrals to obtain
the marginal probability ``\\mathcal{P}[\\bm{x}]`` are performed using the
specified `algorithm`.

Overall, `num_samples` Monte Carlo samples are performed. For each
individual sample, one or mupltiple marginalization operations need to be performed.

If `progress == true`, a progress bar will be shown during the computation.

# Returns

Returns a `DataFrame` containing the results of the simulation. This resulting
`DataFrame` has 3 columns. Assuming, the returned value has been named `result`
the columns can be accessed by:

- `result.MutualInformation`: A vector of vectors that contains the results of the simulation. Each element of the outer vector is the result of a single Monte Carlo sample. Each element is a vector containing the trajectory mutual information estimates for each time specified in `system.dtimes`.
- `result.TimeMarginal`: A vector containing, for each sample, the CPU time in seconds used for the computation of the marginal entropy.
- `result.TimeConditional`: A vector containing, for each sample, the CPU time in seconds used for the computation of the conditional entropy.
"""
function mutual_information(
    system::AbstractSystem,
    algorithm;
    num_samples::Integer=1,
    progress=true,
    threads=false,
    distributed=false,
    compile_args=(;)
)
    # initialize the ensembles
    compiled_system = compile(system; compile_args...)

    # this is the outer Direct Monte-Carlo loop
    result = if distributed
        _mi_inner_distributed(compiled_system, algorithm, num_samples, progress)
    elseif threads
        _mi_inner_multithreaded(compiled_system, algorithm, num_samples, progress)
    else
        _mi_inner(compiled_system, algorithm, num_samples, progress)
    end

    result
end

function _compute(system, algorithm, i, rng; progress=nothing)
    sample = generate_configuration(system; rng=rng)
    # compute ln [P(x,s)/(P(x)P(s))]
    info = @timed information_density(system, algorithm, sample; rng=rng)
    mi, metrics = info.value
    if progress !== nothing
        next!(progress)
    end
    res = to_dataframe(sample)
    res[!, :MutualInformation] = mi
    res[!, :N] .= i
    metrics[!, :N] .= i
    (
        DataFrame(
            N=[i],
            CPUTime=[info.time],
            Algorithm=[name(algorithm)]
        ),
        res,
        metrics
    )
end

function _reduce_results(results)
    meta, res, metrics = reduce(results) do l, r
        meta = vcat(l[1], r[1])
        res = vcat(l[2], r[2])
        metrics = vcat(l[3], r[3])
        meta, res, metrics
    end
    (metadata = meta, result = res, metrics = metrics)
end

function _mi_inner(system, algorithm, num_samples, show_progress)
    rng = Random.default_rng()
    result = @showprogress showspeed=true enabled=show_progress map(1:num_samples) do i
        _compute(system, algorithm, i, rng)
    end
    _reduce_results(result)
end

function _mi_inner_multithreaded(compiled_system, algorithm, num_samples, show_progress)
    # We perform the outer Monte Carlo algorithm using all available threads.
    p = Progress(num_samples; showspeed=false, enabled=show_progress)
    result = Vector{Tuple{DataFrame,DataFrame,DataFrame}}(undef, num_samples)
    @sync for i in 1:num_samples
        new_system = copy(compiled_system)
        Threads.@spawn begin
            rng = Random.TaskLocalRNG()
            result[i] = _compute(new_system, algorithm, i, rng; progress=p)
        end
    end
    _reduce_results(result)
end

function _mi_inner_distributed(system, algorithm, num_samples, show_progress)
    result = @showprogress enabled=show_progress pmap(1:num_samples) do i
        rng = Random.Xoshiro(i)
        new_system = copy(system)
        _compute(new_system, algorithm, i, rng)
    end
    _reduce_results(result)
end
