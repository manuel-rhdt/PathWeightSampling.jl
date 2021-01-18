import GaussianMcmc: chemotaxis_system, ConditionalEnsemble, MarginalEnsemble, marginal_configuration, generate_configuration, collect_samples
using Test
using Statistics

system = chemotaxis_system()
cond_ens = ConditionalEnsemble(system)
marg_ens = MarginalEnsemble(system)
initial = generate_configuration(system)
dtimes = collect(range(0.0, 2.0, length=51)[2:end])

function _logmeanexp(x::AbstractArray)
    x_max = maximum(x)
    log(mean(xi -> exp(xi - x_max), x)) + x_max
end
logmeanexp(x::AbstractArray; dims=nothing) = if dims === nothing _logmeanexp(x) else mapslices(_logmeanexp, x, dims=dims) end

collect_samples(initial, cond_ens, 10, dtimes)
collect_samples(marginal_configuration(initial), marg_ens, 10, dtimes)

N = 10_000
final = map(1:50) do i
    ce = collect_samples(initial, cond_ens, N, dtimes)
    ce = -logmeanexp(ce, dims=2)
    me = collect_samples(marginal_configuration(initial), marg_ens, N, dtimes)
    me = -logmeanexp(me, dims=2)
    vec(me .- ce)
end
final = hcat(final...)

using Plots
 
plot(dtimes, mean(final, dims=2), ribbon=3*std(final, dims=2) ./ sqrt(100))

@test all((mean(final, dims=2) + 3*std(final, dims=2) ./ sqrt(100)) .>= 0.0)

