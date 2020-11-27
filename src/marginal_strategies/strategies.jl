using DataFrames

abstract type SimulationResult end

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

