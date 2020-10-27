include("basic_setup.jl")

samples, acceptance = Trajectories.generate_mcmc_samples(initial, system, 2^8, 2^18)

grid = range(0.0, 500.0, length=100)

vals = map(x->x[1], samples[1](grid))

p = plot(samples[1])
plot!(p, grid, vals)

resampled = map(s -> map(x->x[1], s(grid)), samples)
resampled = hcat(resampled...)

p = plot(grid, mean(resampled, dims=2), ylim=(0, 60))
plot!(p, system.response)