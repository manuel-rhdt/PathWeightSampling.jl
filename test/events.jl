import PWS: collect_trajectory, Thin, MergeWith, sub_trajectory, merge_trajectories
using Test
using StaticArrays

num_samples = 10000
rand_u = [rand([0, 1], 2) for i = 1:num_samples]
rand_t = cumsum(-log.(rand(num_samples)))
rand_i = collect(1:num_samples)
iter = zip(rand_u, rand_t, rand_i)
all_events = collect(iter)

thinned_events = all_events |> Thin() |> collect

for event in thinned_events
    @test event ∈ all_events
end

for (index, (u, t, i)) in enumerate(all_events)
    if index == 1
        continue
    end
    (uprev, tprev, iprev) = all_events[index - 1]
    du = u - uprev
    if all(du .== 0)
        @test tprev ∉ getindex.(thinned_events, 2)
    else
        @test (uprev, tprev, iprev) ∈ thinned_events
    end
end


for i in 1:length(thinned_events) - 2
    @test thinned_events[i][1] != thinned_events[i + 1][1]
end

@test thinned_events[end][1:2] == all_events[end][1:2]

sub1 = sub_trajectory(iter, [1])
sub2 = sub_trajectory(iter, [2])
merg = sub1 |> MergeWith(sub2) |> collect

@test merg == thinned_events

sub1_trim  = collect(sub_trajectory(iter, [1]))[100:110] |> collect_trajectory
a = sub1_trim |> MergeWith(sub2) |> collect
b = sub2 |> MergeWith(sub1_trim) |> collect

@test getindex.(a, 2) == getindex.(b, 2)
for (a, b) in zip(a, b)
    @test a[1][[1,2]]==b[1][[2,1]]
end
