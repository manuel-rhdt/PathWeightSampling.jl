using DataFrames

abstract type SimulationResult end

logmeanexp(x) = log(mean(exp.(x .- maximum(x)))) + maximum(x)

include("ThermodynamicIntegration.jl")
include("AIS.jl")
include("DirectMC.jl")

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

function mutual_information(system::SRXsystem, algorithm; num_responses::Integer=1, kwargs...)
    # initialize the ensembles
    cond_ensemble = ConditionalEnsemble(system)
    marg_ensemble = MarginalEnsemble(system)

    # this is the outer Direct Monte-Carlo loop
    stats, results = Base.invokelatest(_mi_inner, system, cond_ensemble, marg_ensemble, algorithm, num_responses; kwargs...)

    cond_res = getindex.(results, 1)
    marg_res = getindex.(results, 2)

    Dict(
        "mutual_information" => stats, 
        "marginal_entropy_estimate" => summary(marg_res...),
        "conditional_entropy_estimate" => summary(cond_res...)
    )
end

function _mi_inner(system::SRXsystem, cond_ensemble, marg_ensemble, algorithm, num_responses; kwargs...)
    stats = DataFrame(
        MI=zeros(Float64, num_responses),
        MarginalEntropy=zeros(Float64, num_responses),
        ConditionalEntropy=zeros(Float64, num_responses),
        TimeConditional=zeros(Float64, num_responses), 
        TimeMarginal=zeros(Float64, num_responses), 
    )

    result = map(1:num_responses) do i
        # draw an independent sample
        initial = generate_configuration(system)

        # compute P(x|s) and P(x)
        cond_result = @timed simulate(algorithm, initial, cond_ensemble; kwargs...)
        marg_result = @timed simulate(algorithm, marginal_configuration(initial), marg_ensemble; kwargs...)


        # ln [P(x,s)/(P(x)P(s))] = ln [P(x|s)/P(x)] = ln P(x|s) - ln P(x)
        stats.MI[i] = log_marginal(cond_result.value) - log_marginal(marg_result.value)

        stats.MarginalEntropy[i] = log_marginal(marg_result.value)
        stats.ConditionalEntropy[i] = log_marginal(cond_result.value)
        stats.TimeConditional[i] = cond_result.time
        stats.TimeMarginal[i] = marg_result.time

        (cond_result.value, marg_result.value)
    end

    stats, result
end