using DataFrames

abstract type SimulationResult end

function _logmeanexp(x::AbstractArray)
    x_max = maximum(x)
    log(mean(xi -> exp(xi - x_max), x)) + x_max
end
logmeanexp(x::AbstractArray; dims=nothing) = if dims === nothing _logmeanexp(x) else mapslices(_logmeanexp, x, dims=dims) end

include("ThermodynamicIntegration.jl")
include("AIS.jl")
include("DirectMC.jl")
include("SMC.jl")

function marginal_entropy(
    system;
    algorithm,
    num_responses::Int=1,
    kwargs...
)
    stats = DataFrame(
        Sample=zeros(Float64, num_responses), 
        Variance=zeros(Float64, num_responses), 
        TimeElapsed=zeros(Float64, num_responses), 
        GcTime=zeros(Float64, num_responses),
        InitialEnergy=zeros(Float64, num_responses)
    )

    results = map(1:num_responses) do i
        initial = generate_configuration(system)
        stats.InitialEnergy[i] = -energy_difference(initial, system)

        timed_result = @timed simulate(algorithm, initial, system; kwargs...)

        sample = log_marginal(timed_result.value)
        variance = var(timed_result.value)

        stats.Sample[i] = sample
        stats.Variance[i] = variance
        stats.TimeElapsed[i] = timed_result.time
        stats.GcTime[i] = timed_result.gctime

        timed_result.value
    end

    Dict("marginal_entropy" => stats, "marginal_entropy_estimate" => summary(results...))
end


function conditional_entropy(system;  num_responses::Int=1)
    result = zeros(Float64, num_responses)
    for i in 1:num_responses
        conf = generate_configuration(system)
        # result[i] = log P(x|s)
        result[i] = -energy_difference(conf, system)
    end

    Dict("conditional_entropy" => DataFrame(Sample=[mean(result)], NumSamples=[num_responses]))
end

function mutual_information(system, algorithm; num_responses::Integer=1)
    # initialize the ensembles
    compiled_system = compile(system)

    # this is the outer Direct Monte-Carlo loop
    # result = Base.invokelatest(_mi_inner, compiled_system, algorithm, num_responses)
    _mi_inner(compiled_system, algorithm, num_responses)

    result
end

function _mi_inner(compiled_system, algorithm, num_responses)
    stats = DataFrame(
        TimeConditional=zeros(Float64, num_responses), 
        TimeMarginal=zeros(Float64, num_responses), 
    )

    mi = map(1:num_responses) do i
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