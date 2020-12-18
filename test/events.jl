import GaussianMcmc: EventThinner, sub_trajectory, merge_trajectories
using Test
using StaticArrays

num_samples = 10000
rand_u = [SVector{2}(rand([0, 1], 2)) for i=1:num_samples-1]
push!(rand_u, rand_u[end])
rand_t = vcat(0.0, cumsum(-log.(rand(num_samples-1))))
iter = zip(rand_u, rand_t)
all_events = collect(iter)

thinned = EventThinner(iter)
for elem in thinned
    @test elem ∈ all_events
end

thinned_events = [x for x in thinned]
for i in 1:length(thinned_events)-2
    @test thinned_events[i][1] != thinned_events[i+1][1]
end

@test thinned_events[end] == all_events[end]

sub1 = sub_trajectory(iter, SA[1])
sub2 = sub_trajectory(iter, SA[2])
merg = merge_trajectories(sub1, sub2)
@test collect(merg) == thinned_events
