using Test
import PathWeightSampling as PWS

import Random: Xoshiro

rates = [50.0, 1.0]
rstoich = [[], [1 => 1]]
nstoich = [[1 => 1], [1 => -1]]

bd_reactions = PWS.ReactionSet(rates, rstoich, nstoich, [:X, :Y])

u0 = [50.0]
tspan = (0.0, 100.0)

bd_system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    bd_reactions,
    u0,
    tspan,
    :Y,
    :X
)

dtimes = tspan[1]:0.1:tspan[2]
trace = PWS.generate_configuration(bd_system; rng=Xoshiro(1))
agg, traj = PWS.JumpSystem.generate_trajectory(bd_system, dtimes; rng=Xoshiro(1))

begin
    u_arr = [u0]
    for r in trace.rx
        if r == 1
            push!(u_arr, u_arr[end] .+ 1)
        elseif r == 2
            push!(u_arr, u_arr[end] .- 1)
        else
            error("unreachable")
        end
    end
    u_at_time(t) = u_arr[searchsortedfirst(trace.t, t)]
    for (i, t) in enumerate(dtimes)
        @test u_at_time(t) == traj[:, i]
    end
    @test u_arr[end] == agg.u == traj[:, end]
end
