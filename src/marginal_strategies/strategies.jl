using DataFrames
using ProgressMeter

abstract type SimulationResult end


function _logmeanexp(x::AbstractArray)
    x_max = maximum(x)
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

include("ThermodynamicIntegration.jl")
include("AIS.jl")
include("DirectMC.jl")
include("SMC.jl")
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
function mutual_information(system, algorithm; num_samples::Integer=1, progress=true, compile_args=(;))
    # initialize the ensembles
    compiled_system = compile(system; compile_args...)

    # this is the outer Direct Monte-Carlo loop
    # result = Base.invokelatest(_mi_inner, compiled_system, algorithm, num_samples)
    result = _mi_inner(compiled_system, algorithm, num_samples, progress)

    result
end

function _mi_inner(compiled_system, algorithm, num_samples, show_progress)
    stats = DataFrame(
        TimeConditional=zeros(Float64, num_samples),
        TimeMarginal=zeros(Float64, num_samples),
    )

    p = Progress(num_samples; showspeed=true, enabled=show_progress)
    result = progress_map(1:num_samples, progress=p) do i
        # draw an independent sample
        sample = generate_configuration(compiled_system)

        # compute P(x|s) and P(x)
        cond_result = @timed conditional_density(compiled_system, algorithm, sample)
        marg_result = @timed marginal_density(compiled_system, algorithm, sample)

        # record the simulation time
        stats.TimeConditional[i] = cond_result.time
        stats.TimeMarginal[i] = marg_result.time

        # summary of the configuration
        s = summary(sample)

        # compute a sample for the mutual information using
        # ln [P(x,s)/(P(x)P(s))] = ln [P(x|s)/P(x)] = ln P(x|s) - ln P(x)
        cond_result.value - marg_result.value, s
    end

    stats[!, :MutualInformation] = getindex.(result, 1)
    stats[!, :Trajectory] = getindex.(result, 2)

    stats
end