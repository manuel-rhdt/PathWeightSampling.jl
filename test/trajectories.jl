using GaussianMcmc
using Test

using GaussianMcmc.Trajectories

traj = Trajectory([:S, :X], [0.0, 1.0, 2.0], [1 2 3; 4 5 6])

@test length(traj) == 3 == length(traj.t)
@test collect(traj) == [(0.0, [1, 4]), (1.0, [2,5]), (2.0, [3, 6])]

