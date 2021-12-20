using DrWatson
using DataFrames
using Statistics
using CairoMakie
using HDF5
using LaTeXStrings
using CSVFiles
using Dates
using Interpolations

sims = DataFrame(load(projectdir("figures", "figure_chemotaxis", "data", "simulation_2021-10-12.csv")))

plot_theme = Theme(
    font = "Noto Sans",
    fontsize = 8,
    colgap = 4,
    rowgap = 0,
    linewidth = 2,
    markersize = 5,
    spinewidth = 0.5,
    figure_padding = (5, 5, 10, 5),
    Axis = (
        spinewidth = 1.0,
        xgridwidth = 1.0,
        xminorgridwidth = 1.0,
        yminorgridwidth = 1.0,
        ygridwidth = 1.0,
        xtickwidth = 1.0,
        ytickwidth = 1.0,
        xticksize = 3,
        yticksize = 3,
        xlabelpadding = 0,
        ylabelpadding = 0,
        xticklabelpad = 0,
        yticklabelpad = 3,
        # xlabelfont = "DejaVu Sans Condensed",
        # ylabelfont = "DejaVu Sans Condensed",
    ),
    Legend = (
        framevisible = false,
        framewidth = 1.0,
        rowgap = 0,
        titlegap = 2,
        titlefontsize = 11,
        groupgap = 5,
        patchsize = (8, 10),
        padding = (1, 1, 1, 1),
        linewidth = 1.5,
    )
)
set_theme!(plot_theme)

fig = Figure(resolution = (246, 300))
ax1 = Axis(fig[1, 1], xlabel = "traj. duration (s)", title = "mutual\ninformation (bits)")

groups = filter(g -> g.TauL[1] ∈ [0.01, 0.1, 1.0, 10.0], groupby(sims, [:M, :TauL], sort = true))
cmap = cgrad(:OrRd_9)
colors = cmap[3:2:9]

for (g, c) in zip(groups, colors)
    lines!(ax1, g.DiscreteTimes, g.Mean ./ log(2), label = "$(g.TauL[1]) s", color = c)
end
Legend(fig[1, 3], ax1, "input\ncorr.\ntime", orientation = :vertical, tellwidth = true)
colgap!(fig.layout, 2, Fixed(-35))

ax2 = Axis(fig[1, 2], xlabel = "traj. duration (s)", title = "information\n rate (bits/s)")
for (g, c) in zip(groups, colors)
    nodes = g.DiscreteTimes[1]:10*diff(g.DiscreteTimes)[1]:g.DiscreteTimes[end]
    itp = CubicSplineInterpolation(nodes, g.Mean[begin:10:end] ./ log(2))
    us = range(extrema(g.DiscreteTimes)...; step = 2)
    vs = [Interpolations.gradient(itp, u)[1] for u in us]
    lines!(ax2, us, vs, color = c)
end
ylims!(ax2, 0, nothing)

subgl = GridLayout()
fig.layout[2, 1:2] = subgl
# subgl.alignmode = Mixed(right=0)

ax3 = Axis(
    subgl[1, 1],
    title="optimal input timescale",
    xlabel = "input correlation time (s)",
    ylabel = "information rate (bits/s)",
    xscale = log10,
    xminorticks = IntervalsBetween(9),
    xminorgridvisible = true
)
xvals = Float64[]
yvals = Float64[]
for g in groupby(sims, [:M, :TauL], sort = true)
    push!(xvals, g.TauL[1])
    indexfrom = searchsortedfirst(g.DiscreteTimes, 150)
    rates = diff(g.Mean) ./ diff(g.DiscreteTimes) ./ log(2)
    asymptotic_rate = mean(rates[indexfrom:end])
    push!(yvals, asymptotic_rate)
end
ax3.xticks = [0.01, 0.1, 1.0, 10.0]
ax3.xtickformat = "{:.2f}"
lines!(ax3, xvals, yvals, color = :black)
ylims!(ax3, 0, 15)

c4 = :darkseagreen
ax4 = Axis(
    subgl[1, 1], 
    ytickcolor = c4, 
    yticklabelcolor = c4, 
    yaxisposition = :right, 
    xscale = log10, 
    ylabel = "instantaneous\nmutual information (bits)",
    ylabelcolor = c4)
hidespines!(ax4)
hidexdecorations!(ax4)
hideydecorations!(ax4, label=false, ticks=false, ticklabels=false)
imi_data = DataFrame(load(projectdir("figures", "figure_chemotaxis", "data", "imi.csv")))
imi_data_no_methylation = DataFrame(load(projectdir("figures", "figure_chemotaxis", "data", "imi_no_methylation.csv")))
lines!(ax4, imi_data.TauL, imi_data.IMI, color = c4, linewidth=1.5)
lines!(ax4, imi_data_no_methylation.TauL, imi_data_no_methylation.IMI, color = c4, linewidth=1.5, linestyle=:dash)
ylims!(ax4, 0, 1.5)

rowsize!(fig.layout, 2, Auto(0.8))

for (label, layout) in zip(["b", "c", "d"], [fig[1, 1], fig[1, 2], fig[2, 1:3]])
    Label(layout[1, 1, TopLeft()], label,
        textsize = 14,
        font = "Noto Sans Bold",
        padding = (0, 10, 7, 0),
        halign = :right)
end

save(projectdir("figures", "figure_chemotaxis", "chemotaxis_mi.svg"), fig, pt_per_unit = 1)

fig