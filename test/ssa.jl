using Test
import PathWeightSampling as PWS

# create a simple birth-death system
rates = [1.0, 1.0]
rstoich = [[], [1 => 1]]
nstoich = [[1 => 1], [1 => -1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, 1)

agg = PWS.build_aggregator(PWS.GillespieDirect(), reactions, 1:2; seed=1234)

@test agg.rng == Xoshiro(1234)
@test agg.sumrate == 0.0

agg = PWS.initialize_aggregator(agg, reactions)

@test agg.rng != Xoshiro(1234)
@test agg.u == [0]
@test agg.tstop > 0
@test agg.sumrate == 1.0

trace = PWS.ReactionTrace([], [])

agg2 = PWS.step_ssa(agg, reactions, nothing, trace)

@test agg2.tstop > agg.tstop
@test agg2.u == [1]
@test agg2.sumrate == 2.0
@test agg2.weight != 0

@test trace.rx == [1]
@test trace.t == [agg.tstop]

agg3 = agg
for i = 1:100
    agg3 = PWS.step_ssa(agg3, reactions, nothing, trace)
end

trace

@test agg3.u[1] == -sum(2 .* trace.rx .- 3)

agg = PWS.initialize_aggregator(agg, reactions, u0=[0], active_reactions=BitSet())

@test agg.tstop == Inf
@test agg.trace_index == 1

trace_new = PWS.ReactionTrace([], [])

agg = PWS.step_ssa(agg, reactions, trace, trace_new)

@test agg.tstop == Inf
@test agg.u == [1]
@test agg.trace_index == 2

for i = 1:100
    agg = PWS.step_ssa(agg, reactions, trace, trace_new)
    @test agg.tprev == trace.t[i+1]
    @test agg.tstop == Inf
    @test agg.trace_index == i + 2
end

@test trace == trace_new

# test absorbing state

rates = [1.0]
rstoich = [[1 => 1]]
nstoich = [[1 => -1]]
reactions = PWS.ReactionSet(rates, rstoich, nstoich, 1)

agg = PWS.build_aggregator(PWS.GillespieDirect(), reactions, 1:1)
agg = PWS.initialize_aggregator(agg, reactions, tspan=(0.0, 10.0))
@test agg.u == [0]
@test agg.tstop == Inf
agg = PWS.step_ssa(agg, reactions, nothing, nothing)
@test agg.tprev == 10.0
@test agg.weight == 0.0

# test MarkovJumpSystem with coupled birth death processes

rates = [50.0, 1.0, 1.0, 1.0]
rstoich = [[], [1 => 1], [1 => 1], [2 => 1]]
nstoich = [[1 => 1], [1 => -1], [2 => 1], [2 => -1]]

reactions = PWS.ReactionSet(rates, rstoich, nstoich, 2)

system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    [0, 0, 1, 2],
    BitSet([3, 4])
)

agg, trace = PWS.generate_trace(system, [50, 50], (0.0, 10.0))

ce = agg.weight

@time samples = [PWS.sample(trace, system, [50, 50], (0.0, 10.0)) for i = 1:10000]

using Statistics
result_direct = mean(ce .- samples)

system_dep = PWS.MarkovJumpSystem(
    PWS.DepGraphDirect(),
    reactions,
    [0, 0, 1, 2],
    BitSet([3, 4])
)

agg2, trace2 = PWS.generate_trace(system_dep, [50, 50], (0.0, 10.0))

@time samples_dep = [PWS.sample(trace, system_dep, [50, 50], (0.0, 10.0)) for i = 1:10000]

result_dep = mean(ce .- samples_dep)

@test result_direct ≈ result_dep rtol = 0.05


# ChemotaxisJumps

system = PWS.simple_chemotaxis_system(
    n_clusters=800,
    n=6,
    duration=10.0,
    dt=0.1
)
dtimes = system.tspan[1]:system.dt:system.tspan[2]
# @test sum(system.u0[system.reactions.receptors]) == 25

PWS.make_depgraph(system.reactions)

for (rid, gid) in enumerate(system.agg.ridtogroup)
    if gid == 1
        @test PWS.reaction_type(system.reactions, rid)[1] == 2
    elseif gid == 2
        @test PWS.reaction_type(system.reactions, rid)[1] == 3
    else
        @test PWS.reaction_type(system.reactions, rid)[1] < 2
    end
end

conf = PWS.generate_configuration(system, seed=2)

@test conf.trace isa PWS.HybridTrace
@test PWS.ReactionTrace(conf.trace) isa PWS.ReactionTrace

alg = PWS.SMCEstimate(512)

nresults = Threads.nthreads()
me_results = Vector{PWS.SimulationResult}(undef, nresults)
ce_results = Vector{PWS.SimulationResult}(undef, nresults)

@time begin
    Threads.@threads for i = 1:nresults
        me_results[i] = PWS.simulate(alg, PWS.ReactionTrace(conf.trace), system; new_particle=PWS.HybridParticle)
        ce_results[i] = PWS.simulate(alg, conf.trace, system; new_particle=PWS.MarkovParticle)
    end
end

me_marginal = hcat((PWS.log_marginal(r) for r in me_results)...)

begin
f = Figure()
ax = Axis(f[1,1])
for i=1:size(me_marginal, 2)
    lines!(ax, dtimes, me_marginal[:, i])
end
f
end

@time be_result = PWS.simulate(alg, conf.trace, system; new_particle=PWS.BennettParticle)

using CairoMakie
import LinearAlgebra
norm_prob(a) = LinearAlgebra.normalize!(exp.(a .- maximum(a)))
lines(be_result.num_samples[:, 1])

me_normal = mapslices(norm_prob, me_result.log_marginal_estimate; dims=2)
ce_normal = mapslices(norm_prob, ce_result.log_marginal_estimate; dims=2)

sum(ce_result.num_samples, dims=2)[:, 1]
lines(sum(me_normal .* me_result.num_samples, dims=2)[:, 1])

lines(dtimes, PWS.log_marginal(ce_result) - PWS.log_marginal(me_result))
lines(PWS.log_marginal(be_result))

histogram(diff(conf.trace.t))

begin
    input = []
    act = []
    weight = []
    function inspect(new_bag)
        push!(input, [p.agg.u[1] for p in new_bag])
        push!(act, [sum(p.agg.cache.p_a .* p.agg.u[system.reactions.receptors]) / sum(p.agg.u[system.reactions.receptors]) for p in new_bag])
        push!(weight, [p.agg.weight for p in new_bag])
    end
    me_result = PWS.marginal_density(system, alg, conf; inspect=inspect)
    act_mat_me = hcat(act...)
    input_mat_me = hcat(input...)
    weight_mat_me = hcat(weight...)
    input = []
    act = []
    weight = []
    ce_result = PWS.conditional_density(system, alg, conf; inspect=inspect)
    act_mat_ce = hcat(act...)
    input_mat_ce = hcat(input...)
    weight_mat_ce = hcat(weight...)
end

result = me_result
log_marginal = me_result

# ce_result
# Plots.plot(dtimes, ce_result .- me_result)
# Plots.plot(dtimes[2:end], diff(ce_result .- me_result))

cum_weights = cumsum(weight_mat_ce, dims=2)
cum_nweights = exp.(cum_weights .- maximum(cum_weights, dims=1))
cum_ps = cum_nweights ./ sum(cum_nweights, dims=1)
# Plots.plot(dtimes[2:end], (cum_ps)', legend=false)

test_time = searchsortedfirst(dtimes, 1.2)

using CairoMakie



using Statistics
nweight_mat = exp.(weight_mat_me .- maximum(weight_mat_me, dims=1))
pweight_mat = nweight_mat ./ sum(nweight_mat, dims=1)
Plots.scatter(dmat, pweight_mat, legend=false, markerstrokewidth=0, markersize=1.0, markercolor=:black)

Plots.scatter(dmat, weight_mat_me, markercolor=:blue, legend=false)
Plots.scatter(dmat, weight_mat_me .- mean(weight_mat_ce, dims=1), markercolor=:red, legend=false)

i = 43
dmat[1, i]
histogram(nweight_mat[:, i], bins=20)
nweight_mat[:, i]
Plots.scatter(dmat[:, i-1:i+1], act_mat_me[:, i-1:i+1], markersize=4, markerstrokewidth=0, marker_z=nweight_mat[:, i], c=:algae)

p = begin
    import Plots
    Plots.plot(legend=false)
    Plots.scatter!(dmat, act_mat_me .- 0.33, legend=false, markerstrokewidth=0, markersize=1, marker_z=nweight_mat, c=:algae)
    # Plots.scatter!(dmat, (input_mat_me .- 100) ./ 10, markersize=0.5, markerstrokewidth=0, marker_z=nweight_mat, c=:algae)
    Plots.plot!(dtimes[2:end], (diff(log_marginal) .- maximum(diff(log_marginal))) / 80000, color=:red)
    Plots.plot!(dtimes, (conf.traj[end-1, :] .- 1600) ./ 5000)
    Plots.plot!(dtimes, (conf.traj[1, :] .- 100) ./ 10)
    Plots.ylims!((-0.5, 0.2))
end
savefig(p, "_research/plot.pdf")

prob_mat = exp.(weights_mat .- maximum(weights_mat, dims=1))

prob_mat[:, 5]

using Plots
histogram(prob_mat[:, 5], bins=10)

@run PWS.conditional_density(system, alg, conf)
ce_smc = [PWS.conditional_density(system, alg, conf)[end] for i = 1:10]
me_smc = [PWS.marginal_density(system, alg, conf)[end] for i = 1:10]


@test conf.traj[1, 2:end] == conf.trace.u

@run agg, trace = PWS.generate_trace(system)
agg.cache
@time agg, trace = PWS.generate_trace(system)

agg.jump_search_order

alg = SMCEstimate(128)

@time mi = PWS.mutual_information(system, alg)
mi.MutualInformation
mi.Trajectory