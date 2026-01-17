"""
    trajectories.jl

Tests for trajectory configuration generation.
Validates that configurations (traces and trajectories) are correctly
generated from stochastic simulation and properly indexed with discrete times.
"""

using Test
import PathWeightSampling as PWS

import Random: Xoshiro

@testset "Birth-Death Process Configuration" begin
    rates = [50.0, 1.0]
    rstoich = [Pair{Int, Int}[], [1 => 1]]
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

    dtimes = PWS.discrete_times(bd_system)
    conf = PWS.generate_configuration(bd_system; rng=Xoshiro(1))
    trace = conf.trace
    traj = conf.traj

    @testset "Configuration Generation" begin
        @test conf.discrete_times == dtimes "Discrete times should match system specification"
    end

    @testset "Trajectory Consistency" begin
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
            @test u_at_time(t) == traj[:, i] "Trajectory state should match computed state at time $t"
        end
        @test u_arr[end] == traj[:, end] "Final trajectory state should match simulation"
    end
end
