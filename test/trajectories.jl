module TrajectoryTests

using GaussianMcmc
using Test

using GaussianMcmc.Trajectories

traj = Trajectory([:S], [0.0, 1.0, 2.0], [1 2 3; 4 5 6])

@test length(traj) == 3

end