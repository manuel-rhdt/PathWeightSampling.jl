using Transducers
using DiffEqJump
using StochasticDiffEq
import ModelingToolkit: SDESystem, get_states

struct MarginalEnsemble{JP,U0}
    jump_problem::JP
    dist::TrajectoryDistribution
    dtimes::Vector{Float64}

    # Handling the initial condition. Can be either a vector that specifies the exact initial
    # condition, or an EmpiricalDistribution specifying a probability distribution over initial
    # conditions.
    u0::U0
end

struct ConditionalEnsemble{JP<:DiffEqBase.AbstractJumpProblem,IX,IM}
    jump_problem::JP
    dist::TrajectoryDistribution
    indep_idxs::IX
    index_map::IM
    dtimes::Vector{Float64}
end

struct SXconfiguration{uTs,uTx,tType}
    s_traj::Trajectory{uTs,tType}
    x_traj::Trajectory{uTx,tType}
end

Base.copy(c::SXconfiguration) = SXconfiguration(copy(c.s_traj), copy(c.x_traj))
struct SRXconfiguration{uTs,Utr,Utx,tType}
    s_traj::Trajectory{uTs,tType}
    r_traj::Trajectory{Utr,tType}
    x_traj::Trajectory{Utx,tType}
end

Base.copy(c::SRXconfiguration) = SRXconfiguration(copy(c.s_traj), copy(c.r_traj), copy(c.x_traj))

ensurevec(a::AbstractVector) = a
ensurevec(a) = SVector(a)

function initial_log_likelihood(ensemble::ConditionalEnsemble, u0, x_traj)
    0.0
end

function initial_log_likelihood(ensemble::MarginalEnsemble, u0::AbstractVector, x_traj::Trajectory)
    s0 = u0[1]
    x0 = x_traj.u[1][1]
    if ensemble.u0 isa EmpiricalDistribution
        # compute log P(x0 | s0) = log P(s0, x0) - log ∑ₓ P(s0, x)
        logpdf(ensemble.u0, [s0, x0]) - log(sum(x -> pdf(ensemble.u0, [s0, x]), ensemble.u0.axes[2]))
    else
        0.0
    end
end

function Base.getindex(conf::SXconfiguration, indices)
    collect_sub_trajectory(merge_trajectories(conf.s_traj, conf.x_traj), ensurevec(indices))
end

function Base.getindex(conf::SRXconfiguration, indices)
    collect_sub_trajectory(merge_trajectories(conf.s_traj, conf.r_traj, conf.x_traj), ensurevec(indices))
end

sample_initial_condition(u0::AbstractVector) = copy(u0)
sample_initial_condition(u0::EmpiricalDistribution) = rand(u0)

sample_initial_condition(ens::MarginalEnsemble) = ens.u0 isa EmpiricalDistribution ? rand(ens.u0)[[1]] : ens.u0
sample_initial_condition(ens::ConditionalEnsemble) = ens.jump_problem.prob.u0

abstract type JumpNetwork end

"""
    SimpleSystem(input_network, output_network, u0, ps, px, dtimes)

A `SimpleSystem` is a system used to compute the mutual information when
there are no latent variables that need to be integrated out.

A `SimpleSystem` consists of two of ModelingToolkit's `ReactionSystem`s:
One that generates the input trajectories (`input_network`), and one that generates the
output trajectories (`output_network`).

`u0` represents the initial condition of the system.

`ps` and `px` are the parameter vectors of the input and output systems, respectively.

# Examples

Create a `SimpleSystem` from a set of coupled birth-death processes.

```jldoctest
using PathWeightSampling, Catalyst, StaticArrays

sn = @reaction_network begin
    κ, ∅ --> 2L
    λ, L --> ∅
end κ λ

xn = @reaction_network begin
    ρ, L --> L + 2X
    μ, X --> ∅
end ρ μ

u0 = SA[10, 20]
dtimes = 0:0.5:10.0
ps = [5.0, 1.0]
px = [3.0, 0.1]

system = PathWeightSampling.SimpleSystem(sn, xn, u0, ps, px, dtimes)

# output

SimpleSystem with 4 reactions
Input variables: L(t)
Output variables: X(t)
Initial condition:
    L(t) = 10
    X(t) = 20
Parameters:
    κ = 5.0
    λ = 1.0
    ρ = 3.0
    μ = 0.1
```

"""
struct SimpleSystem <: JumpNetwork
    sn::ReactionSystem
    xn::ReactionSystem

    u0::Union{AbstractVector,EmpiricalDistribution}

    ps::AbstractVector
    px::AbstractVector

    dtimes

    jump_problem::DiffEqBase.AbstractJumpProblem
    dist::TrajectoryDistribution
end

function SimpleSystem(sn, xn, u0, ps, px, dtimes, dist=nothing; aggregator=Direct())
    joint = ModelingToolkit.extend(xn, sn)

    tp = (first(dtimes), last(dtimes))
    p = vcat(ps, px)
    dprob = DiscreteProblem(joint, sample_initial_condition(u0), tp, p)
    jprob = JumpProblem(convert(ModelingToolkit.JumpSystem, joint), dprob, aggregator, save_positions=(false, false))

    if dist === nothing
        update_map = build_update_map(joint, xn)
        dist = distribution(joint, p; update_map)
    end

    SimpleSystem(sn, xn, u0, ps, px, dtimes, jprob, dist)
end

function Base.show(io::IO, ::MIME"text/plain", system::SimpleSystem)
    joint = reaction_network(system)
    print(io, "SimpleSystem with ", Catalyst.numreactions(joint), " reactions\nInput variables: ")
    ivars = independent_species(system.sn)
    print(io, ivars[1])
    for i = 2:length(ivars)
        print(io, ", ", ivars[i])
    end
    print(io, "\nOutput variables: ")
    ovars = independent_species(system.xn)
    print(io, ovars[1])
    for i = 2:length(ovars)
        print(io, ", ", ovars[i])
    end
    print(io, "\nInitial condition:")
    if system.u0 isa AbstractVector
        jvars = Catalyst.species(joint)
        for i in eachindex(system.u0)
            print(io, "\n    ", jvars[i], " = ", system.u0[i])
        end
    else
        print(io, "\n", system.u0)
    end
    print(io, "\nParameters:")
    p_names = Catalyst.reactionparams(system.sn)
    for i in eachindex(p_names)
        print(io, "\n    ", p_names[i], " = ", system.ps[i])
    end
    p_names = Catalyst.reactionparams(system.xn)
    for i in eachindex(p_names)
        print(io, "\n    ", p_names[i], " = ", system.px[i])
    end
end

struct CompiledSimpleSystem{JP}
    system::SimpleSystem
    marginal_ensemble::MarginalEnsemble{JP}
end

compile(s::SimpleSystem; aggregator=Direct()) = CompiledSimpleSystem(s, MarginalEnsemble(s; aggregator))
marginal_density(csx::CompiledSimpleSystem, algorithm, conf::SXconfiguration) = log_marginal(simulate(algorithm, conf, csx.marginal_ensemble))
conditional_density(csx::CompiledSimpleSystem, algorithm, conf::SXconfiguration) = -energy_difference(conf, csx.marginal_ensemble)

"""
    ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes, dist=nothing; aggregator=Direct())

A `ComplexSystem` is a system used to compute the mutual information with 
latent variables that need to be integrated out.

Therefore, a `ComplexSystem` consists of three ModelingToolkit `ReactionSystem`s:
One that generates the input trajectories (`sn`), one that models the latent variables (`rn`),
and one that generates the output trajectories (`xn`).

# Examples

```jldoctest
using PathWeightSampling, Catalyst, StaticArrays

sn = @reaction_network begin
    κ, ∅ --> 2L
    λ, L --> ∅
end κ λ

rn = @reaction_network begin
    ρ, L + R --> L + LR
    μ, LR --> R
    ξ, R + CheY --> R + CheYp
    ν, CheYp --> CheY
end ρ μ ξ ν

xn = @reaction_network begin
    δ, CheYp --> CheYp + X
    χ, X --> ∅
end δ χ

u0 = SA[10, 30, 0, 50, 0, 0]
dtimes = 0:0.5:10.0
ps = [5.0, 1.0]
pr = [1.0, 4.0, 1.0, 2.0]
px = [1.0, 1.0]

system = PathWeightSampling.ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes)

# output

ComplexSystem with 8 reactions
Input variables: L(t)
Latent variables: R(t), LR(t), CheY(t), CheYp(t)
Output variables: X(t)
Initial condition:
    L(t) = 10
    R(t) = 30
    LR(t) = 0
    CheY(t) = 50
    CheYp(t) = 0
    X(t) = 0
Parameters:
    κ = 5.0
    λ = 1.0
    ρ = 1.0
    μ = 4.0
    ξ = 1.0
    ν = 2.0
    δ = 1.0
    χ = 1.0
```

"""
struct ComplexSystem <: JumpNetwork
    sn::ReactionSystem
    rn::ReactionSystem
    xn::ReactionSystem

    u0::Union{AbstractVector,EmpiricalDistribution}

    ps::AbstractVector
    pr::AbstractVector
    px::AbstractVector

    dtimes

    jump_problem
    dist::TrajectoryDistribution
end

function ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes, dist=nothing; aggregator=Direct())
    joint = ModelingToolkit.extend(xn, ModelingToolkit.extend(rn, sn))

    tp = (first(dtimes), last(dtimes))
    p = vcat(ps, pr, px)
    dprob = DiscreteProblem(joint, sample_initial_condition(u0), tp, p)
    jprob = JumpProblem(convert(ModelingToolkit.JumpSystem, joint), dprob, aggregator, save_positions=(false, false))

    if dist === nothing
        update_map = build_update_map(joint, xn)
        dist = distribution(joint, p; update_map)
    end

    ComplexSystem(sn, rn, xn, u0, ps, pr, px, dtimes, jprob, dist)
end

function Base.show(io::IO, ::MIME"text/plain", system::ComplexSystem)
    joint = reaction_network(system)
    print(io, "ComplexSystem with ", Catalyst.numreactions(joint), " reactions\nInput variables: ")
    ivars = independent_species(system.sn)
    print(io, ivars[1])
    for i = 2:length(ivars)
        print(io, ", ", ivars[i])
    end
    print(io, "\nLatent variables: ")
    lvars = independent_species(system.rn)
    print(io, lvars[1])
    for i = 2:length(lvars)
        print(io, ", ", lvars[i])
    end
    print(io, "\nOutput variables: ")
    ovars = independent_species(system.xn)
    print(io, ovars[1])
    for i = 2:length(ovars)
        print(io, ", ", ovars[i])
    end
    print(io, "\nInitial condition:")
    if system.u0 isa AbstractVector
        jvars = Catalyst.species(joint)
        for i in eachindex(system.u0)
            print(io, "\n    ", jvars[i], " = ", system.u0[i])
        end
    else
        print(io, "\n", system.u0)
    end
    print(io, "\nParameters:")
    p_names = Catalyst.reactionparams(system.sn)
    for i in eachindex(p_names)
        print(io, "\n    ", p_names[i], " = ", system.ps[i])
    end
    p_names = Catalyst.reactionparams(system.rn)
    for i in eachindex(p_names)
        print(io, "\n    ", p_names[i], " = ", system.pr[i])
    end
    p_names = Catalyst.reactionparams(system.xn)
    for i in eachindex(p_names)
        print(io, "\n    ", p_names[i], " = ", system.px[i])
    end
end


"""
# Examples

```jldoctest
using PathWeightSampling, Catalyst, StaticArrays

@parameters α σ
@variables t L(t)

D = Differential(t)

eqs = [D(L) ~ -α * L]
noiseeqs = [σ*L]
@named sn = SDESystem(eqs, noiseeqs, t, [L], [α, σ])

rn = @reaction_network begin
    ρ, L + R --> L + LR
    μ, LR --> R
    ξ, R + CheY --> R + CheYp
    ν, CheYp --> CheY
end ρ μ ξ ν

xn = @reaction_network begin
    δ, CheYp --> CheYp + X
    χ, X --> ∅
end δ χ

u0 = SA[10, 30, 0, 50, 0, 0]
dtimes = 0:0.5:10.0
ps = [5.0, 1.0]
pr = [1.0, 4.0, 1.0, 2.0]
px = [1.0, 1.0]

system = PathWeightSampling.SDEDrivenSystem(sn, rn, xn, u0, ps, pr, px, dtimes)

# output

SDEDrivenSystem with 6 reactions
Input variables: L(t)
Latent variables: R(t), LR(t), CheY(t), CheYp(t)
Output variables: X(t)
Initial condition:
    L(t) = 10
    R(t) = 30
    LR(t) = 0
    CheY(t) = 50
    CheYp(t) = 0
    X(t) = 0
Parameters:
    α = 5.0
    σ = 1.0
    ρ = 1.0
    μ = 4.0
    ξ = 1.0
    ν = 2.0
    δ = 1.0
    χ = 1.0
```
"""
struct SDEDrivenSystem <: JumpNetwork
    sn::SDESystem
    rn::ReactionSystem
    xn::ReactionSystem

    u0::Union{AbstractVector,EmpiricalDistribution}

    ps::AbstractVector
    pr::AbstractVector
    px::AbstractVector

    dtimes

    jump_problem
    dist::TrajectoryDistribution
end

function SDEDrivenSystem(sn, rn, xn, u0, ps, pr, px, dtimes, dist=nothing; aggregator=Direct())
    joint = ModelingToolkit.extend(xn, rn)

    tp = (first(dtimes), last(dtimes))
    p = vcat(ps, pr, px)

    # Make sure we don't use integers with SDEs
    if u0 isa AbstractVector
        u0 = map(Float64, u0)
    end

    init = sample_initial_condition(u0)
    @assert length(init) == length(get_states(joint))
    dprob = DiscreteProblem(joint, init, tp, vcat(pr, px))
    jprob = JumpProblem(joint, dprob, aggregator, save_positions=(false, false))

    if dist === nothing
        update_map = build_update_map(joint, xn)
        dist = distribution(joint, p; update_map)
    end

    SDEDrivenSystem(sn, rn, xn, u0, ps, pr, px, dtimes, jprob, dist)
end

function Base.show(io::IO, ::MIME"text/plain", system::SDEDrivenSystem)
    joint = reaction_network(system)
    print(io, "SDEDrivenSystem with ", Catalyst.numreactions(joint), " reactions\nInput variables: ")
    ivars = independent_species(system.sn)
    print(io, ivars[1])
    for i = 2:length(ivars)
        print(io, ", ", ivars[i])
    end
    print(io, "\nLatent variables: ")
    lvars = independent_species(system.rn)
    print(io, lvars[1])
    for i = 2:length(lvars)
        print(io, ", ", lvars[i])
    end
    print(io, "\nOutput variables: ")
    ovars = independent_species(system.xn)
    print(io, ovars[1])
    for i = 2:length(ovars)
        print(io, ", ", ovars[i])
    end
    print(io, "\nInitial condition:")
    if system.u0 isa AbstractVector
        jvars = Catalyst.species(joint)
        for i in eachindex(system.u0)
            print(io, "\n    ", jvars[i], " = ", system.u0[i])
        end
    else
        print(io, "\n", system.u0)
    end
    print(io, "\nParameters:")
    p_names = system.sn.ps
    for i in eachindex(p_names)
        print(io, "\n    ", p_names[i], " = ", system.ps[i])
    end
    p_names = Catalyst.reactionparams(system.rn)
    for i in eachindex(p_names)
        print(io, "\n    ", p_names[i], " = ", system.pr[i])
    end
    p_names = Catalyst.reactionparams(system.xn)
    for i in eachindex(p_names)
        print(io, "\n    ", p_names[i], " = ", system.px[i])
    end
end

struct CompiledComplexSystem{Sys,JP,JPC,IXC,DXC}
    system::Sys
    marginal_ensemble::MarginalEnsemble{JP}
    conditional_ensemble::ConditionalEnsemble{JPC,IXC,DXC}
end

compile(s::Union{ComplexSystem,SDEDrivenSystem}; marginal_aggregator=Direct(), conditional_aggregator=Direct()) = CompiledComplexSystem(s, MarginalEnsemble(s; aggregator=marginal_aggregator), ConditionalEnsemble(s; aggregator=conditional_aggregator))
marginal_density(csrx::CompiledComplexSystem, algorithm, conf::Union{SXconfiguration,SRXconfiguration}) = log_marginal(simulate(algorithm, marginal_configuration(conf), csrx.marginal_ensemble))
conditional_density(csrx::CompiledComplexSystem, algorithm, conf::Union{SXconfiguration,SRXconfiguration}) = log_marginal(simulate(algorithm, conf, csrx.conditional_ensemble))


tspan(sys::JumpNetwork) = (Float64(first(sys.dtimes)), Float64(last(sys.dtimes)))

reaction_network(system::SimpleSystem) = ModelingToolkit.extend(system.xn, system.sn)
reaction_network(system::ComplexSystem) = ModelingToolkit.extend(system.xn, ModelingToolkit.extend(system.rn, system.sn))
reaction_network(system::SDEDrivenSystem) = ModelingToolkit.extend(system.xn, system.rn)

function _solve(system::SimpleSystem)
    sol = solve(system.jump_problem, SSAStepper())
end


function start_collect(::Nothing, indices)
    # return empty trajectory
    Trajectory(Vector{Int16}[], Float64[], Int[])
end

function start_collect(((u, t, i), state)::Tuple, indices)
    Trajectory([u[indices]], typeof(t)[], typeof(i)[])
end

function step_collect!(result::Trajectory, ::Nothing, (uprev, tprev, iprev)::Tuple, indices)
    # end trajectory
    push!(result.t, tprev)
    push!(result.i, iprev)
    result
end

function step_collect!(
    result::Trajectory,
    ((u, t, i), state)::Tuple,
    (uprev, tprev, iprev)::Tuple,
    indices
)
    if (@view u[indices]) != result.u[end]
        push!(result.u, u[indices])
        push!(result.t, tprev)
        push!(result.i, iprev)
    end
    result
end

function collect_sub_trajectory(iter, indices)
    f = iterate(iter)
    traj = start_collect(f, indices)
    while f !== nothing
        val, state = f
        f = iterate(iter, state)
        traj = step_collect!(traj, f, val, indices)
    end
    traj
end

function collect_sub_trajectories(iter, indices_list...)
    f = iterate(iter)
    trajs = map(indices -> start_collect(f, indices), indices_list)
    while f !== nothing
        val, state = f
        f = iterate(iter, state)
        foreach((traj, indices) -> step_collect!(traj, f, val, indices), trajs, indices_list)
    end
    trajs
end

function generate_configuration(system::Union{SimpleSystem,ComplexSystem}; seed=rand(UInt))
    joint = reaction_network(system)
    u0 = SVector(map(Int16, sample_initial_condition(system.u0))...)
    jp = remake(system.jump_problem, u0=u0)

    s_spec = independent_species(system.sn)
    s_idxs = SVector(species_indices(joint, s_spec)...)
    x_spec = independent_species(system.xn)
    x_idxs = SVector(species_indices(joint, x_spec)...)

    @assert s_idxs == collect(1:length(s_idxs))
    @assert x_idxs == collect(length(u0)-length(x_idxs)+1:length(u0))

    iter = SSAIter(init(jp, SSAStepper(), tstops=(), seed=seed))

    s_traj, x_traj = collect_sub_trajectories(iter, s_idxs, x_idxs)

    SXconfiguration(s_traj, x_traj)
end

function generate_configuration(system::SDEDrivenSystem; seed=rand(UInt))
    joint = reaction_network(system)

    u0 = SVector(map(Float64, sample_initial_condition(system.u0))...)
    jp = remake(system.jump_problem, u0=u0)

    s_spec = independent_species(system.sn)
    s_idxs = SVector(species_indices(joint, s_spec)...)
    x_spec = independent_species(system.xn)
    x_idxs = SVector(species_indices(joint, x_spec)...)

    @assert s_idxs == collect(1:length(s_idxs))
    @assert x_idxs == collect(length(u0)-length(x_idxs)+1:length(u0))

    u0s::Vector{Float64} = [x for x in u0[s_idxs]]
    s_prob = SDEProblem(system.sn, u0s, tspan(system), system.ps)
    sol = solve(s_prob, SOSRA(), saveat=0.01)
    input_traj = Trajectory(sol.u[1:end-1], sol.t[2:end])
    djp = DrivenJumpProblem(jp, input_traj)
    iter = SSAIter(init(djp; tstops=()))

    s_traj, x_traj = collect_sub_trajectories(iter, s_idxs, x_idxs)

    SXconfiguration(s_traj, x_traj)
end

function _solve(system::ComplexSystem)
    sol = solve(system.jump_problem, SSAStepper())
end

function generate_full_configuration(system::ComplexSystem; seed=rand(UInt))
    joint = reaction_network(system)
    u0 = SVector(map(Int16, sample_initial_condition(system.u0))...)
    jp = remake(system.jump_problem, u0=u0)

    s_spec = independent_species(system.sn)
    s_idxs = SVector(species_indices(joint, s_spec)...)

    r_spec = independent_species(system.rn)
    r_idxs = SVector(species_indices(joint, r_spec)...)

    x_spec = independent_species(system.xn)
    x_idxs = SVector(species_indices(joint, x_spec)...)

    @assert vcat(s_idxs, r_idxs, x_idxs) == collect(eachindex(u0))

    iter = SSAIter(init(jp, SSAStepper(), tstops=(), seed=seed))

    s_traj, r_traj, x_traj = collect_sub_trajectories(iter, s_idxs, r_idxs, x_idxs)

    SRXconfiguration(s_traj, r_traj, x_traj)
end

function build_update_map(joint::ReactionSystem, xn::ReactionSystem)
    update_map = Int[]
    spmap = Catalyst.speciesmap(joint)
    mapper = (x, y)::Pair -> spmap[x] => y

    unique_netstoich = unique(map(r -> mapper.(r.netstoich), Catalyst.reactions(xn)))

    for react in Catalyst.reactions(joint)
        new_index = 0
        for (k, un) in enumerate(unique_netstoich)
            if mapper.(react.netstoich) == un
                new_index = k
                break
            end
        end

        push!(update_map, new_index)
    end
    update_map
end

function MarginalEnsemble(system::SimpleSystem; aggregator=Direct())
    joint = reaction_network(system)
    s_idxs = species_indices(joint, Catalyst.species(system.sn))

    dprob = DiscreteProblem(system.sn, SVector(sample_initial_condition(system.u0)[s_idxs]...), tspan(system), system.ps)
    jprob = JumpProblem(convert(ModelingToolkit.JumpSystem, system.sn), dprob, aggregator, save_positions=(false, false))

    u0 = system.u0 isa EmpiricalDistribution ? system.u0 : SVector(system.u0[s_idxs]...)
    MarginalEnsemble(jprob, system.dist, collect(system.dtimes), u0)
end

function MarginalEnsemble(system::ComplexSystem; aggregator=Direct())
    sr_network = ModelingToolkit.extend(system.rn, system.sn)
    joint = reaction_network(system)
    sr_idxs = species_indices(joint, Catalyst.species(sr_network))

    # check that our assumptions are correct
    @assert sr_idxs == collect(1:Catalyst.numspecies(sr_network))

    u0 = sample_initial_condition(system.u0)[sr_idxs]
    dprob = DiscreteProblem(sr_network, SVector(u0...), tspan(system), vcat(system.ps, system.pr))
    jprob = JumpProblem(sr_network, dprob, aggregator, save_positions=(false, false))

    MarginalEnsemble(jprob, system.dist, collect(system.dtimes), SVector(u0...))
end

function MarginalEnsemble(system::SDEDrivenSystem; aggregator=Direct())
    s_idxs = collect(1:length(get_states(system.sn)))
    sr_idxs = collect(1:Catalyst.numspecies(system.rn))

    u0 = sample_initial_condition(system.u0)
    u0s::Vector{Float64} = [x for x in u0[s_idxs]]
    s_prob = SDEProblem(system.sn, u0s, tspan(system), system.ps)

    u0r = map(Float64, u0[sr_idxs])
    dprob = DiscreteProblem(system.rn, SVector(u0r...), tspan(system), system.pr)
    jprob = JumpProblem(system.rn, dprob, aggregator, save_positions=(false, false))

    driven_jump_problem = DrivenJumpProblem(jprob, s_prob)

    MarginalEnsemble(driven_jump_problem, system.dist, collect(system.dtimes), SVector(u0r...))
end


function ConditionalEnsemble(system::Union{ComplexSystem,SDEDrivenSystem}; aggregator=Direct())
    joint = reaction_network(system)
    # we need the `system.rn` network indices to find the correct initial condition U0
    # for this network
    r_idxs = species_indices(joint, Catalyst.species(system.rn))

    # check that our assumptions are correct
    @assert r_idxs == collect(1:Catalyst.numspecies(system.rn))

    u0 = sample_initial_condition(system.u0)[r_idxs]
    dprob = DiscreteProblem(system.rn, SVector(u0...), (first(system.dtimes), last(system.dtimes)), system.pr)
    jprob = JumpProblem(system.rn, dprob, aggregator, save_positions=(false, false))

    indep_species = independent_species(system.rn)
    indep_idxs = SVector(species_indices(system.rn, indep_species)...)

    s_species = get_states(system.sn)
    index_map = SVector(species_indices(system.rn, s_species)...)

    ConditionalEnsemble(jprob, system.dist, indep_idxs, index_map, collect(system.dtimes))
end

function species_indices(rs::ReactionSystem, species)
    getindex.(Ref(Catalyst.speciesmap(rs)), species)
end

independent_species(sde::SDESystem) = get_states(sde)
function independent_species(rs::ReactionSystem)
    i_spec = []
    smap = Catalyst.speciesmap(rs)
    for r in Catalyst.reactions(rs)
        push!(i_spec, getindex.(r.netstoich, 1)...)
    end
    sort(unique(s for s ∈ i_spec), by=x -> smap[x])
end

function dependent_species(rs::ReactionSystem)
    smap = Catalyst.speciesmap(rs)
    sort(setdiff(Catalyst.species(rs), independent_species(rs)), by=x -> smap[x])
end

function create_integrator(conf::SXconfiguration, ensemble::MarginalEnsemble, u0, tspan::Tuple)
    if ensemble.jump_problem isa DrivenJumpProblem
        driven_jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
        init(driven_jprob)
    else
        jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
        init(jprob, SSAStepper(), tstops=(), save_start=false, save_end=false)
    end
end

function sample(configuration::SXconfiguration, system::MarginalEnsemble; θ=0.0)
    if θ != 0.0
        error("can only use DirectMC with JumpNetwork")
    end
    u0 = SVector(sample_initial_condition(system)...)
    tspan = (system.dtimes[begin], system.dtimes[end])
    integrator = create_integrator(configuration, system, u0, tspan)
    iter = SSAIter(integrator)
    s_traj = collect_trajectory(iter)
    SXconfiguration(s_traj, configuration.x_traj)
end

function collect_samples(initial::SXconfiguration, ensemble::MarginalEnsemble, num_samples::Int)
    result = Array{Float64,2}(undef, length(ensemble.dtimes), num_samples)
    for result_col ∈ eachcol(result)
        u0 = sample_initial_condition(ensemble)
        tspan = (ensemble.dtimes[begin], ensemble.dtimes[end])
        integrator = create_integrator(initial, ensemble, u0, tspan)
        iter = SSAIter(integrator) |> Map((u, t, i)::Tuple -> (u, t, 0))
        cumulative_logpdf!(result_col, ensemble.dist, merge_trajectories(iter, initial.x_traj), ensemble.dtimes)
        result_col .+= initial_log_likelihood(ensemble, u0, initial.x_traj)
    end

    result
end

function propagate(conf::SXconfiguration, ensemble::MarginalEnsemble, u0, tspan::Tuple)
    integrator = create_integrator(conf, ensemble, u0, tspan)
    iter = SSAIter(integrator)
    ix1 = searchsortedfirst(conf.x_traj.t, tspan[1])
    merged = merge_trajectories(iter, Base.Iterators.rest(conf.x_traj, ix1))

    # TODO: check species ordering
    log_weight = trajectory_energy(ensemble.dist, merged, tspan=tspan)

    copy(integrator.u), log_weight
end

function energy_difference(configuration::SXconfiguration, ensemble::MarginalEnsemble)
    log_p0 = initial_log_likelihood(ensemble, configuration.s_traj.u[1], configuration.x_traj)
    -cumulative_logpdf(ensemble.dist, configuration.s_traj |> MergeWith(configuration.x_traj), ensemble.dtimes) .- log_p0
end

function sample(configuration::Union{SXconfiguration,SRXconfiguration}, ensemble::ConditionalEnsemble; θ=0.0)
    if θ != 0.0
        error("can only use DirectMC with JumpNetwork")
    end
    integrator = create_integrator(configuration, ensemble, ensemble.jump_problem.prob.u0, ensemble.jump_problem.prob.tspan)
    iter = SSAIter(integrator)
    rtraj = collect_sub_trajectory(iter, ensemble.indep_idxs)
    SRXconfiguration(configuration.s_traj, rtraj, configuration.x_traj)
end

function collect_samples(initial::Union{SXconfiguration,SRXconfiguration}, ensemble::ConditionalEnsemble, num_samples::Int)
    driven_jp = DrivenJumpProblem(ensemble.jump_problem, initial.s_traj)

    result = Array{Float64,2}(undef, length(ensemble.dtimes), num_samples)
    for result_col ∈ eachcol(result)
        integrator = init(driven_jp)
        iter = SSAIter(integrator) |> Map((u, t, i)::Tuple -> (u, t, 0))
        cumulative_logpdf!(result_col, ensemble.dist, merge_trajectories(iter, initial.x_traj), ensemble.dtimes)
    end

    result
end

function simulate(algorithm::DirectMCEstimate, initial::Union{SXconfiguration,SRXconfiguration}, system)
    samples = collect_samples(initial, system, algorithm.num_samples)
    DirectMCResult(samples)
end

function create_integrator(conf::Union{SXconfiguration,SRXconfiguration}, ensemble::ConditionalEnsemble, u0, tspan::Tuple)
    jprob = remake(ensemble.jump_problem, u0=u0, tspan=tspan)
    driven_jp = DrivenJumpProblem(jprob, conf.s_traj, index_map=ensemble.index_map)
    init(driven_jp)
end

# propagate the initial condition forward in time and compute the corresponding increase in log-likelihood
function propagate(conf::Union{SXconfiguration,SRXconfiguration}, ensemble::ConditionalEnsemble, u0, tspan::Tuple)
    integrator = create_integrator(conf, ensemble, u0, tspan)
    iter = SSAIter(integrator)
    ix1 = searchsortedfirst(conf.x_traj.t, tspan[1])

    merged = merge_trajectories(iter, Base.Iterators.rest(conf.x_traj, ix1))

    # TODO: check species ordering
    log_weight = trajectory_energy(ensemble.dist, merged, tspan=tspan)

    copy(integrator.u), log_weight
end

function energy_difference(configuration::SRXconfiguration, ensemble::ConditionalEnsemble)
    traj = merge_trajectories(configuration.s_traj, configuration.r_traj, configuration.x_traj)
    -cumulative_logpdf(ensemble.dist, traj, ensemble.dtimes)
end

marginal_configuration(conf::SXconfiguration) = conf

function marginal_configuration(conf::SRXconfiguration)
    new_s = merge_trajectories(conf.s_traj, conf.r_traj)
    SXconfiguration(collect_trajectory(new_s, nocopy=true), conf.x_traj)
end

# MCMC Moves in trajectory space

mutable struct TrajectoryChain{Ensemble} <: MarkovChain
    ensemble::Ensemble
    # interaction parameter
    θ::Float64

    # to save statistics
    last_regrowth::Float64
    accepted_list::Vector{Float64}
    rejected_list::Vector{Float64}
end

chain(ensemble; θ::Real=1.0) = TrajectoryChain(ensemble, θ, 0.0, Float64[], Float64[])

# reset statistics
function reset(pot::TrajectoryChain)
    resize!(pot.accepted_list, 0)
    resize!(pot.rejected_list, 0)
end

function accept(pot::TrajectoryChain)
    push!(pot.accepted_list, pot.last_regrowth)
end

function reject(pot::TrajectoryChain)
    push!(pot.rejected_list, pot.last_regrowth)
end

function energy(conf::SXconfiguration, chain::TrajectoryChain; θ=chain.θ)
    if θ > zero(θ)
        -θ * trajectory_energy(chain.ensemble.dist, conf.s_traj |> MergeWith(conf.x_traj))
    else
        0.0
    end
end

function energy(conf::SRXconfiguration, chain::TrajectoryChain; θ=chain.θ)
    if θ > zero(θ)
        traj = merge_trajectories(conf.s_traj, conf.r_traj, conf.x_traj)
        θ * logpdf(chain.ensemble.dist, traj)
    else
        0.0
    end
end

function propose!(new_conf, old_conf, chain::TrajectoryChain)
    chain.last_regrowth = propose!(new_conf, old_conf, chain.ensemble)
    new_conf
end
propose!(new_conf::SXconfiguration, old_conf::SXconfiguration, ensemble::MarginalEnsemble) = propose!(new_conf.s_traj, old_conf.s_traj, ensemble)

function propose!(new_traj::Trajectory, old_traj::Trajectory, ensemble::MarginalEnsemble)
    jump_problem = ensemble.jump_problem

    regrow_duration = rand() * duration(old_traj)

    if rand(Bool)
        shoot_forward!(new_traj, old_traj, jump_problem, old_traj.t[end] - regrow_duration)
    else
        shoot_backward!(new_traj, old_traj, jump_problem, old_traj.t[begin] + regrow_duration)
    end

    regrow_duration
end

function shoot_forward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::DiffEqBase.AbstractJumpProblem, branch_time::Real)
    branch_value = old_traj(branch_time)
    branch_point = searchsortedfirst(old_traj.t, branch_time)
    tspan = (branch_time, old_traj.t[end])

    empty!(new_traj.u)
    empty!(new_traj.t)
    empty!(new_traj.i)
    append!(new_traj.u, @view old_traj.u[begin:branch_point-1])
    append!(new_traj.t, @view old_traj.t[begin:branch_point-1])
    append!(new_traj.i, @view old_traj.i[begin:branch_point-1])

    jump_problem = DiffEqBase.remake(jump_problem; u0=branch_value, tspan=tspan)
    integrator = DiffEqBase.init(jump_problem, SSAStepper())
    iter = SSAIter(integrator)
    new_branch = collect_trajectory(iter)

    append!(new_traj.t, new_branch.t)
    append!(new_traj.u, new_branch.u)
    append!(new_traj.i, new_branch.i)
    nothing
end

function shoot_backward!(new_traj::Trajectory, old_traj::Trajectory, jump_problem::DiffEqBase.AbstractJumpProblem, branch_time::Real)
    branch_value = old_traj(branch_time)
    branch_point = searchsortedfirst(old_traj.t, branch_time)
    tspan = (old_traj.t[begin], branch_time)

    jump_problem = DiffEqBase.remake(jump_problem; u0=branch_value, tspan=tspan)
    integrator = DiffEqBase.init(jump_problem, SSAStepper())
    iter = SSAIter(integrator)
    new_branch = collect_trajectory(iter)

    empty!(new_traj.u)
    empty!(new_traj.t)
    empty!(new_traj.i)

    append!(new_traj.u, @view new_branch.u[end-1:-1:begin])
    append!(new_traj.u, @view old_traj.u[branch_point:end])
    append!(new_traj.i, @view new_branch.i[end-1:-1:begin])
    append!(new_traj.i, @view old_traj.i[branch_point:end])

    for rtime in @view new_branch.t[end:-1:begin+1]
        push!(new_traj.t, branch_time - rtime)
    end
    append!(new_traj.t, @view old_traj.t[branch_point:end])
    nothing
end

# ========
# PLOTTING
# ========

@recipe function f(conf::SXconfiguration)
    @series begin
        label := "input"
        conf.s_traj
    end
    @series begin
        label := "output"
        conf.x_traj
    end
end
