using DrWatson
using DataFrames
using Statistics
using CairoMakie
using LaTeXStrings
using CSVFiles

normalfont = projectdir("fonts", "NotoSans-Condensed.ttf")
italicfont = projectdir("fonts", "NotoSans-Italic.ttf")
boldfont = projectdir("fonts", "NotoSans-CondensedSemiBold.ttf")
subfigurelabelfont = projectdir("fonts", "NotoSans-Bold.ttf")

plot_theme = Theme(
    font = normalfont,
    fontsize = 8,
    colgap = 8,
    rowgap = 4,
    linewidth = 1.0,
    markersize = 8,
    spinewidth = 0.5,
    figure_padding = (7, 7, 4, 4),
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
        ylabelpadding = 2,
        xticklabelpad = 0,
        yticklabelpad = 1,
    ),
    Legend = (
        framevisible = false,
        framewidth = 1.0,
        rowgap = 0,
        titlegap = 0,
        titlefont = boldfont,
        titlehalign = :left,
        halign = :center,
        valign = :top,
        gridshalign = :left,
        titlefontsize = 10,
        patchlabelgap = 3,
        groupgap = 3,
        patchsize = (10, 5),
        padding = (1, 1, 1, 1),
        margin = (0, 0, 0, 0),
        linewidth = 1.25,
    )
)

# plot_theme = Theme(
#     font = normalfont,
#     fontsize = 48,
#     colgap = 4,
#     rowgap = 0,
#     linewidth = 5,
#     markersize = 20,
#     figure_padding = (2, 2, 2, 2),
#     Axis = (
#         spinewidth = 3.0,
#         xgridwidth = 3.0,
#         xminorgridwidth = 2.0,
#         yminorgridwidth = 2.0,
#         ygridwidth = 3.0,
#         xtickwidth = 3.0,
#         ytickwidth = 3.0,
#         xticksize = 10,
#         yticksize = 10,
#         xlabelpadding = 0,
#         ylabelpadding = 2,
#         xticklabelpad = 0,
#         yticklabelpad = 1,
#     ),
#     Legend = (
#         framevisible = false,
#         framewidth = 1.0,
#         rowgap = 2,
#         titlegap = 2,
#         titlefont = boldfont,
#         titlehalign = :left,
#         halign = :center,
#         gridshalign = :left,
#         titlefontsize = 11,
#         groupgap = 5,
#         patchsize = (10, 5),
#         padding = (1, 1, 1, 1),
#         linewidth = 1.5,
#     )
# )

set_theme!(plot_theme)

colors = Dict(
    "Mattingly" => :green,
    "Literature" => :red,
    "Fit" => :blue
)

fig = Figure(resolution = (246, 320))
# fig = Figure(resolution = (1280, 800))


axb = Axis(
    fig[2,1],
    xticksvisible=false,
    xticklabelsvisible=false,
    yticksvisible=false,
    yticklabelsvisible=false,
    xgridvisible=false,
    ygridvisible=false,
    xlabel=L"x",
    title="diffusion in\nconcentration gradient"
)

# poly!(axb,
#     [(0, 0), (0, 1), (1, 1), (1, 0)], 
#     color = [:white, :white, :gray, :gray]
# )

import CairoMakie: Gray, GrayA

gauss(x, σ) = 0.06/(sqrt(2π)*σ) * exp(-(x-0.5)^2/σ^2)
x = 0:1e-2:1
b1 = band!(axb, x, 0, gauss.(x, Ref(0.06)), color = (:gray20, 0.7), label=L"\mathrm{P}(x; t=1)")
b2 = band!(axb, x, 0, gauss.(x, Ref(0.15)), linestyle=:dot, color = (:gray40, 0.7), linewidth=1.5, label=L"\mathrm{P}(x; t=2)")
b3 = band!(axb, x, 0, gauss.(x, Ref(0.3)), linestyle=:dot, color = (:gray70, 0.7), linewidth=1.5, label=L"\mathrm{P}(x; t=3)")
translate!(b1, (0, 0, 2)) # move to the foreground
translate!(b2, (0, 0, 1)) # move to the foreground

lines!(axb, x, exp.(x .- 3).+0.2, label=L"c(x)=c_0e^{gx}")

# band!(x, 0, gauss.(x, Ref(0.08)), color = GrayA(0.5, 0.7))
# band!(x, 0, gauss.(x, Ref(0.2)), color = GrayA(0.7, 0.7))
    
xlims!(axb, 0, 1)
ylims!(axb, 0, 1)

axislegend(axb, position=(0, 1), margin=(2,2,1,1))

axc = Axis(fig[2,2], title="input signal", ylabel=L"c(t)", xlabel=L"t")

input_data = DataFrame(load(projectdir("figure_rates", "data", "input_traj_summary.csv")))

band!(axc, input_data.t, input_data.q05, input_data.q95, color=:gray70)
band!(axc, input_data.t, input_data.q25, input_data.q75, color=:gray50)
for i=[1,2,4, 5]
    lines!(axc, input_data.t, input_data[:, Symbol(:traj, i)], linewidth=1)
end

xlims!(axc, 0, 100)

axd = Axis(
    fig[3, 1], 
    xlabel = "traj. duration T (s)",
    ylabel = L"I(\mathbf{C}; \mathrm{\mathbf{Y_p}})",
    title = "mutual\ninformation (bits)"
)

sims = DataFrame(load(projectdir("figure_rates", "data", "2023-02-17_mi_trajectories.csv")))

groups = groupby(sims, [:g], sort = true)
cmap = cgrad(:OrRd_9, rev=true)
ngroups = length(groups)
colors = cmap[range(0, 0.7, length=ngroups)]

for (g, c) in zip(groups, colors)
    lines!(axd, g.DiscreteTimes, g.Mean ./ log(2), label = "$(g.g[1] * 1e3)", color = c)
end

Legend(
    fig[4, 1:2], 
    axd, 
    "gradient steepness (mm⁻¹)", 
    orientation = :horizontal, 
    tellwidth = true,
    titlehalign = :center
)
# colgap!(fig.layout, 2, Fixed(-35))

axe = Axis(
    fig[3, 2],
    xlabel = "traj. duration T (s)",
    ylabel = L"I(\mathbf{C}; \mathrm{\mathbf{Y_p}})/T",
    title = "information\nrate (bits/s)"
)

for (g, c) in zip(groups, colors)
    t = g.DiscreteTimes[2:end]
    mi = g.Mean ./ log(2)
    lines!(axe, t, mi[2:end] ./ t, label = "$(g.g[1] * 1e3)", color = c)
end

# for (g, c) in zip(groups, colors)
#     nodes = g.DiscreteTimes[1]:10*diff(g.DiscreteTimes)[1]:g.DiscreteTimes[end]
#     itp = CubicSplineInterpolation(nodes, g.Mean[begin:10:end] ./ log(2))
#     us = range(extrema(g.DiscreteTimes)...; step = 2)
#     vs = [Interpolations.gradient(itp, u)[1] for u in us]
#     lines!(axe, us, vs, color = c)
# end
ylims!(axe, nothing, 0.16)

save(projectdir("figure_rates", "chemotaxis.pdf"), fig, pt_per_unit = 1)
save(projectdir("figure_rates", "chemotaxis.png"), fig, pt_per_unit = 1)


fig