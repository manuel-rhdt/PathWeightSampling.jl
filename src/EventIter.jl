import DiffEqJump

struct SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}
    integrator::DiffEqJump.SSAIntegrator{F,uType,tType,P,S,CB,SA,OPT,TS}
end

Base.IteratorSize(::Type{SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}}) where {F,uType,tType,P,S,CB,SA,OPT,TS} = Base.SizeUnknown()
Base.eltype(::Type{SSAIter{F,uType,tType,P,S,CB,SA,OPT,TS}}) where {F,uType,tType,P,S,CB,SA,OPT,TS} = Tuple{uType,tType,Int}

function Base.iterate(iter::SSAIter)
    integrator = iter.integrator
    end_time = integrator.sol.prob.tspan[2]
    if integrator.t >= end_time
        return nothing
    end
    aggregator = integrator.cb.condition
    i = aggregator.next_jump

    next_jump_time = integrator.tstop > integrator.t ? integrator.tstop : typemax(integrator.tstop)
    if !isempty(integrator.tstops) &&
       integrator.tstops_idx <= length(integrator.tstops) &&
       integrator.tstops[integrator.tstops_idx] < next_jump_time
        next_jump_time = integrator.tstops[integrator.tstops_idx]
        i = 0
    end
    t = min(next_jump_time, end_time)
    if t == end_time
        i = 0
    end
    return (integrator.u, t, i), ()
end

function Base.iterate(iter::SSAIter, state::Tuple{})
    integrator = iter.integrator
    if DiffEqJump.should_continue_solve(integrator)
        end_time = integrator.sol.prob.tspan[2]
        step!(integrator)
        aggregator = integrator.cb.condition
        i = aggregator.next_jump

        next_jump_time = integrator.tstop > integrator.t ? integrator.tstop : typemax(integrator.tstop)
        if !isempty(integrator.tstops) &&
           integrator.tstops_idx <= length(integrator.tstops) &&
           integrator.tstops[integrator.tstops_idx] < next_jump_time
            next_jump_time = integrator.tstops[integrator.tstops_idx]
            i = 0
        end
        t = min(next_jump_time, end_time)
        if t == end_time
            i = 0
        end

        return (integrator.u, t, i), ()
    end
    nothing
end

SciMLBase.getsyms(iter::SSAIter) = SciMLBase.getsyms(iter.integrator.sol)
SciMLBase.getindepsym(iter::SSAIter) = SciMLBase.getindepsym(iter.integrator.sol)