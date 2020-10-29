include("basic_setup.jl")

using Plots
using Statistics
using LaTeXStrings

grid_size = 100
num_samples = 2^18
θs = range(0,1,length=6)

resampled_samples = zeros((grid_size, num_samples, length(θs)))
for (i, θ) in enumerate(θs)
    println("θ=$θ")
    system.θ = θ
    samples, acceptance = Trajectories.generate_mcmc_samples(initial, system, 2^8, num_samples)
    grid = range(0.0, 500.0, length=grid_size)

    for (j, s) in enumerate(samples)
        for (k, x) in enumerate(s(grid))
            resampled_samples[k, j, i] = x[1]
        end
    end
end

x = range(0.0, 500.0, length=grid_size)
y = reshape(mean(resampled_samples, dims=2), (length(x), :))
yerror = reshape(std(resampled_samples, dims=2), (length(x), :))
yerror[:,1:5] .= 0.0

pyplot()
using Plots.PlotMeasures
p1 = plot(x, y, 
    ribbon=yerror, 
    fillalpha=0.5, 
    line_z=θs', 
    fill_z=θs', 
    label=hcat(("$θ" for θ in θs)...), 
    ylim=(35,65),
    linewidth=2.5,
    c=:linear_blue_5_95_c73_n256,
    ylabel=L"S_t",
    xlabel=L"t",
    title="signals",
    legend=false,
    legendtitle=L"\theta",
    colorbar=false,
)

p2 = plot(system.response, c=:green, title="response", legend=:false)

p = plot(p2, p1, layout=Plots.grid(2,1), size=(250,600))
pos = p.o.axes[3].get_position()
pos.y1 *= 0.5
p.o.axes[3].set_position(pos)
PyPlot.show(p.o)
p.o

savefig(p, projectdir("plots", "signal_example.pdf"))