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
logmeanexp(x::AbstractArray; dims=nothing) = if dims === nothing _logmeanexp(x) else mapslices(_logmeanexp, x, dims=dims) end

include("ThermodynamicIntegration.jl")
include("AIS.jl")
include("DirectMC.jl")
include("SMC.jl")

function mutual_information(system, algorithm; num_responses::Integer=1, progress=true)
    # initialize the ensembles
    compiled_system = compile(system)

    # this is the outer Direct Monte-Carlo loop
    # result = Base.invokelatest(_mi_inner, compiled_system, algorithm, num_responses)
    result = _mi_inner(compiled_system, algorithm, num_responses, progress)

    result
end

function _mi_inner(compiled_system, algorithm, num_responses, show_progress)
    stats = DataFrame(
        TimeConditional=zeros(Float64, num_responses), 
        TimeMarginal=zeros(Float64, num_responses), 
    )

    p = Progress(num_responses; showspeed=true, enabled=show_progress)
    mi = progress_map(1:num_responses, progress=p) do i
        # draw an independent sample
        sample = generate_configuration(compiled_system.system)

        # compute P(x|s) and P(x)
        cond_result = @timed conditional_density(compiled_system, algorithm, sample)
        marg_result = @timed marginal_density(compiled_system, algorithm, sample)

        # record the simulation time
        stats.TimeConditional[i] = cond_result.time
        stats.TimeMarginal[i] = marg_result.time

        # compute a sample for the mutual information using
        # ln [P(x,s)/(P(x)P(s))] = ln [P(x|s)/P(x)] = ln P(x|s) - ln P(x)
        cond_result.value - marg_result.value
    end

    stats[!, :MutualInformation] = mi

    stats
end