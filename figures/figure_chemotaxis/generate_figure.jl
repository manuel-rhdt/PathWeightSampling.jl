using DrWatson
using DataFrames
using Statistics
using CairoMakie
using LaTeXStrings
using CSVFiles
using Dates
using Interpolations

sims = DataFrame(load(projectdir("figures", "figure_chemotaxis", "data", "simulation_2021-10-12.csv")))
sims_wo_methylation = DataFrame(load(projectdir("figures", "figure_chemotaxis", "data", "simulation_2022-01-27.csv")))
imi_data = DataFrame(load(projectdir("figures", "figure_chemotaxis", "data", "imi.csv")))
imi_data_no_methylation = DataFrame(load(projectdir("figures", "figure_chemotaxis", "data", "imi_no_methylation.csv")))
gaus_data = DataFrame(load(projectdir("figures", "figure_chemotaxis", "data", "gaussian_rate.csv")))

normalfont = projectdir("figures", "fonts", "NotoSans-Condensed.ttf")
italicfont = projectdir("figures", "fonts", "NotoSans-Italic.ttf")
boldfont = projectdir("figures", "fonts", "NotoSans-CondensedSemiBold.ttf")
subfigurelabelfont = projectdir("figures", "fonts", "NotoSans-Bold.ttf")

plot_theme = Theme(
    font = normalfont,
    fontsize = 8,
    colgap = 4,
    rowgap = 0,
    linewidth = 2,
    markersize = 5,
    spinewidth = 0.5,
    figure_padding = (2, 2, 2, 2),
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
        yticklabelpad = 1,
    ),
    Legend = (
        framevisible = false,
        framewidth = 1.0,
        rowgap = 2,
        titlegap = 2,
        titlefont = boldfont,
        titlehalign = :left,
        halign = :center,
        gridshalign = :left,
        titlefontsize = 11,
        groupgap = 5,
        patchsize = (10, 5),
        padding = (1, 1, 1, 1),
        linewidth = 1.5,
    )
)
set_theme!(plot_theme)

fig = Figure(resolution = (246, 240))
ax1 = Axis(fig[1, 1], xlabel = "traj. duration (s)", title = "mutual\ninformation (bits)")

groups = filter(g -> g.TauL[1] ∈ [0.01, 0.1, 1.0, 10.0], groupby(sims, [:M, :TauL], sort = true))
cmap = cgrad(:OrRd_9)
colors = cmap[3:2:9]

for (g, c) in zip(groups, colors)
    lines!(ax1, g.DiscreteTimes, g.Mean ./ log(2), label = "$(g.TauL[1]) s", color = c)
end
Legend(fig[1, 3], ax1, "input\ncorrelation\ntime", orientation = :vertical, tellwidth = true)
# colgap!(fig.layout, 2, Fixed(-35))

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
rowsize!(fig.layout, 2, Relative(0.7))

ax3 = Axis(
    subgl[1, 1],
    title = "optimal input timescale",
    ylabel = "bits/s",
    xscale = log10,
    xminorticks = IntervalsBetween(9),
    xminorgridvisible = true,
    titlevisible = false
)

lines!(ax3, gaus_data.TauL, gaus_data.InfoRate, color = "#2b8cbe", label = "Gaussian")

function get_info_rates(sims)
    xvals = Float64[]
    yvals = Float64[]
    for g in groupby(sims, [:M, :TauL], sort = true)
        push!(xvals, g.TauL[1])
        indexfrom = searchsortedfirst(g.DiscreteTimes, 150)
        rates = diff(g.Mean) ./ diff(g.DiscreteTimes) ./ log(2)
        asymptotic_rate = mean(rates[indexfrom:end])
        push!(yvals, asymptotic_rate)
    end
    xvals, yvals
end

xvals, yvals = get_info_rates(sims)
lines!(ax3, xvals, yvals, color = :black, label = "PWS")

xvals, yvals = get_info_rates(sims_wo_methylation)
lines!(ax3, xvals, yvals, color = :black, linestyle = :dot, label = "push-pull")
xlims!(ax3, 1e-2, 10)
ylims!(ax3, 0, 25)
text!(ax3, "information transmission rate R", position = Point(1.1e-2, 20), textsize = 8, align = (:left, :baseline), font = boldfont)

c4 = :black
ax4 = Axis(
    subgl[2, 1],
    xminorticks = IntervalsBetween(9),
    xminorgridvisible = true,
    ytickcolor = c4,
    yticklabelcolor = c4,
    yaxisposition = :left,
    xscale = log10,
    ylabel = "bits",
    xlabel = L"input correlation time $\tau_L$ (s)",
    ylabelcolor = c4)
text!(ax4, "instantaneous mutual information", position = Point(1.1e-2, 1.15), textsize = 8, align = (:left, :baseline), font = boldfont)

# ax4.xticks = [0.01, 0.1, 1.0, 10.0]
# ax4.xtickformat = "{:.2f}"
hidexdecorations!(ax3, grid = false, minorgrid = false)
hidespines!(ax4, :t)

imi_line = lines!(ax4, imi_data.TauL, imi_data.IMI, color = c4, label = "chemotaxis")
translate!(imi_line, 0, 0, 9) # bring to foreground
lines!(ax4, gaus_data.TauL, gaus_data.InstantaneousInfo, color = "#2b8cbe", label = "Gaussian\n(LNA)")
lines!(ax4, imi_data_no_methylation.TauL, imi_data_no_methylation.IMI, color = c4, linestyle = :dot, label = "push-pull")
xlims!(ax4, 1e-2, 10)
ylims!(ax4, 0, 1.49)
# annotations!(ax4, ["no methylation"], [Point2(4, 1.7)], font = boldfont, align = (:right, :center), textsize = 7, rotation = 0.3)

Legend(fig[2, 3], ax4, "systems", valign=0.2, orientation = :vertical, tellwidth = true)

for (label, layout) in zip(["b", "c", "d"], [fig[1, 1], fig[1, 2], fig[2, 1:3]])
    Label(layout[1, 1, TopLeft()], label,
        textsize = 14,
        font = subfigurelabelfont,
        padding = (3, 5, 4, -10),
        halign = :left)
end

# create inset

inset_ax = Axis(fig[2, 1:2],
    width = Relative(0.25),
    height = Relative(0.25),
    titlevisible = false,
    # xminorgridvisible = true,
    # xminorticksvisible = true,
    yminorgridvisible = true,
    xticklabelsize = 7,
    yticklabelsize = 7,
    xlabelpadding = -2,
    ylabelpadding = -2,
    halign = 1.0,
    valign = 1.0,
    xtickalign = 1.0,
    ytickalign = 1.0,
    backgroundcolor = :white,
    xticklabelsvisible = true,
    xgridvisible = true,
    yticklabelsvisible = true,
    xscale = log10,
    xlabel = L"\tau_L",
    ylabel = L"R \tau_L",
    xticks = ([0.01, 10.0], ["0.01", "10"]),
    # xminorticks = 0.01:0.1:10.0,
    yminorticks = IntervalsBetween(4),
    topspinevisible = false,
    rightspinevisible = false,
    yticks = [0, 20],
    xticklabelspace = 0,
    yticklabelspace = 0,
    xlabelfont = italicfont,
    ylabelfont = italicfont
)

xlims!(inset_ax, 1e-2, 1e1)
ylims!(inset_ax, 0, 20)
xvals, yvals = get_info_rates(sims)
lines!(inset_ax, xvals, yvals .* xvals, color = :black, label = "PWS")

translate!(inset_ax.scene, 0, 0, 10)
translate!(inset_ax.elements[:background], 0, 0, 9)
translate!(inset_ax.elements[:xgridlines], 0, 0, 9)
translate!(inset_ax.elements[:ygridlines], 0, 0, 9)
translate!(inset_ax.elements[:xminorgridlines], 0, 0, 9)
translate!(inset_ax.elements[:yminorgridlines], 0, 0, 9)

save(projectdir("figures", "figure_chemotaxis", "chemotaxis_mi.svg"), fig, pt_per_unit = 1)
save(projectdir("figures", "figure_chemotaxis", "chemotaxis_mi.pdf"), fig, pt_per_unit = 1)

fig

# presentation
# f = Figure(resolution = (300, 150))
# ax1 = Axis(f[1, 1], xscale = log10)
# lines!(ax1, 1 ./ imi_data_no_methylation.TauL, imi_data_no_methylation.IMI * 10, color = c4, linestyle = :dot)
# xvals, yvals = get_info_rates(sims_wo_methylation)
# lines!(ax1, 1 ./ xvals, yvals, color = :black)
# save(projectdir("figures", "figure_chemotaxis", "chemotaxis_pres.pdf"), f, pt_per_unit = 1)

# f

# fnew = Figure(resolution = (300, 200))
# ax = Axis(
#     fnew[1, 1],
#     xscale = log10,
#     xminorticks = IntervalsBetween(9),
#     xminorgridvisible = true,
#     xlabel = "input timescale",
#     ylabel = "information accumulated per run (bits)"
# )
# ax.xticks = [0.01, 0.1, 1.0, 10.0]
# ax.xtickformat = "{:.2f}"
# xvals, yvals = get_info_rates(sims)
# lines!(ax, xvals, yvals .* xvals, color = :black, label = "PWS")
# xvals, yvals = get_info_rates(sims_wo_methylation)
# lines!(ax, xvals, yvals .* xvals, color = :black, linestyle = :dot, label = "push-pull")
# lines!(ax, gaus_data.TauL, gaus_data.InfoRate .* gaus_data.TauL, color = "#2b8cbe", label = "Gaussian")

# save(projectdir("figures", "figure_chemotaxis", "info_per_run.pdf"), fnew, pt_per_unit = 1)
# fnew