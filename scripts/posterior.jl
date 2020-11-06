include("basic_setup.jl")

using Plots
using Statistics
using LaTeXStrings

grid_size = 100
num_samples = 2^18
θs = range(0,1,length=3)

t_s = 100
t_x = 10

κ = mean_s / t_s
λ = 1 / t_s
ρ = 1 / t_x
μ = 1 / t_x
mean_x = mean_s * ρ / μ

gen = Trajectories.configuration_generator(sn, rn, [κ, λ], [ρ, μ], mean_s, mean_x)

system, initial = Trajectories.generate_configuration(gen, duration=100.0)

p = plot(initial)
plot!(p, system.response)

resampled_samples = zeros((grid_size, num_samples, length(θs)))
for (i, θ) in enumerate(θs)
    println("θ=$θ")
    chain = Trajectories.chain(system, θ)
    samples, acceptance = Trajectories.generate_mcmc_samples(initial, chain, 2^8, num_samples)
    grid = range(0.0, 100.0, length=grid_size)

    for (j, s) in enumerate(samples)
        for (k, x) in enumerate(s(grid))
            resampled_samples[k, j, i] = x[1]
        end
    end
end

x = range(0.0, 100.0, length=grid_size)
y = reshape(mean(resampled_samples, dims=2), (length(x), :))
yerror = reshape(std(resampled_samples, dims=2), (length(x), :))

pyplot()
using Plots.PlotMeasures
p1 = plot(x, y[:,end],
    ribbon=yerror[:,end], 
    fillalpha=0.2, 
    fill_z=1.0, 
    ylim=(35,65),
    linewidth=0,
    label=nothing,
    c=:viridis,
)
plot!(p1, initial, linewidth=0.9, label="orig. signal")
plot!(p1, x, y,
    fillalpha=0.4, 
    line_z=θs', 
    fill_z=θs', 
    label=hcat(("θ=$θ" for θ in θs)...), 
    linewidth=2.5,
    ylabel=L"S_t",
    xlabel=L"t",
    title="signals",
    legend=true,
    colorbar=false,
    c=:viridis,
)


p2 = plot(system.response, c=:green, title="response", legend=:false, ylabel=L"X_t",ylim=(35,65))

p = plot(p2, p1, layout=Plots.grid(2,1), size=(600,600))
pos = p.o.axes[3].get_position()
pos.y1 *= 0.5
p.o.axes[3].set_position(pos)
PyPlot.show(p.o)
p.o

savefig(p, projectdir("plots", "signal_example.pdf"))