using DataFrames
using ProgressMeter
import Statistics: mean

abstract type AbstractSystem end

discrete_times(::AbstractSystem) = error("Custom system does not implement required function `discrete_times`.")
generate_configuration(::AbstractSystem) = error("Custom system does not implement required function `generate_configuration`.")
conditional_density(::AbstractSystem, algorithm, configuration) = error("Custom system does not implement required function `conditional_density`.")
marginal_density(::AbstractSystem, algorithm, configuration) = error("Custom system does not implement required function `marginal_density`.")
compile(sys::AbstractSystem) = sys

function information_density(s::AbstractSystem, algorithm, configuration)
    cond = conditional_density(s, algorithm, configuration)
    marg = marginal_density(s, algorithm, configuration)

    # ln [P(x,s)/(P(x)P(s))] = ln [P(x|s)/P(x)] = ln P(x|s) - ln P(x)
    cond - marg
end

abstract type SimulationResult end

log_marginal(::SimulationResult) = error("Custom subtype of SimulationResult does not implement required function `log_marginal`.")

abstract type AbstractSimulationAlgorithm end

simulate(s::AbstractSimulationAlgorithm, args...) = error("Unknown simulation algorihm", s)
name(x::AbstractSimulationAlgorithm) = string(typeof(x))

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
function mutual_information(system::AbstractSystem, algorithm; num_samples::Integer=1, progress=true, compile_args=(;))
    # initialize the ensembles
    compiled_system = compile(system; compile_args...)

    # this is the outer Direct Monte-Carlo loop
    result = _mi_inner(compiled_system, algorithm, num_samples, progress)

    result
end

function _mi_inner(compiled_system, algorithm, num_samples, show_progress)
    p = Progress(num_samples; showspeed=false, enabled=show_progress)
    tasks = Vector{Task}(undef, num_samples)
    for i in 1:num_samples
        new_system = copy(compiled_system)
        tasks[i] = Threads.@spawn begin
            sample = generate_configuration(new_system)
            # compute ln [P(x,s)/(P(x)P(s))]
            result = @timed information_density(compiled_system, algorithm, sample)
            next!(p)
            DataFrame(N=i, CPUTime=result.time, MutualInformation=[result.value]), DataFrame(sample, N=i)
        end
    end
    result = fetch.(tasks)

    result, traj = reduce(result) do l, r
        result = vcat(l[1], r[1])
        traj = vcat(l[2], r[2])
        result, traj
    end
    Dict("mutual_information" => result, "trajectories" => conf)
end