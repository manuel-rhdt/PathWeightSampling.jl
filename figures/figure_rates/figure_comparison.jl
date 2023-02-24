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

palette = cgrad(:Egypt, categorical=true)
# cgrad(:glasbey_bw_minc_20_maxl_70_n256, categorical=true)

color_mattingly = palette[1]
color_fit = palette[2]
color_lit = palette[3]

plot_theme = Theme(
    font = normalfont,
    fontsize = 8,
    colgap = 8,
    rowgap = 6,
    linewidth = 2,
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
        padding = (0, 0, 0, 0),
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

colors = [color_lit, color_fit, color_mattingly]
labels = ["literature model", "fitted model", "experiments"]

fig = Figure(resolution = (246, 220))
# fig = Figure(resolution = (1280, 800))

axa = Axis(fig[1,1], title="response kernel", xlabel=L"$t$ (s)", ylabel=L"K(t)")

kern_data_lit = DataFrame(load(projectdir("figure_rates", "data", "kern_data_lit.csv")))
kern_data_fit = DataFrame(load(projectdir("figure_rates", "data", "kern_data_fit.csv")))
t_lit, kern_lit, var_kern_lit = eachcol(kern_data_lit[:, [:t, :Kernel, :Var]]);
t_fit, kern_fit, var_kern_fit = eachcol(kern_data_fit[:, [:t, :Kernel, :Var]]);
Δkern_lit = sqrt.(var_kern_lit) ./ sqrt(length(var_kern_lit));
Δkern_fit = sqrt.(var_kern_fit) ./ sqrt(length(var_kern_fit));

band!(axa, t_lit, kern_lit - Δkern_lit, kern_lit + Δkern_lit, color=(color_lit, 0.5))
lines!(axa, vcat(-10, t_lit), vcat(0, kern_lit), color=color_lit)

band!(axa, t_fit, kern_fit - Δkern_fit, kern_fit + Δkern_fit, color=(color_fit, 0.5))
lines!(axa, vcat(-10, t_fit), vcat(0, kern_fit), color=color_fit)

t = vcat(-10, range(t_lit[begin], t_lit[end], length=2000))
theta(x) = if x >= 0 1 else 0 end
mattingly_kernel(t; G, τ1, τ2) = @. theta(t) * G * (1 - exp(-t/τ1)) * exp(-t/τ2)

band!(axa, t, mattingly_kernel(t; G=1.73 + 0.03, τ1=0.22, τ2=9.90 + 0.3), mattingly_kernel(t; G=1.73 - 0.03, τ1=0.22 - 0.01, τ2=9.90 - 0.3), color=(color_mattingly, 0.5))
lines!(axa, t, mattingly_kernel(t; G=1.73, τ1=0.22, τ2=9.90), color=color_mattingly)

Makie.xlims!(axa, -2, 20)

axb = Axis(fig[1,2], title="noise autocorrelation", xlabel=L"$t$ (s)", ylabel=L"N(t)")

cov_data_lit = DataFrame(load(projectdir("figure_rates", "data", "cov_data_lit.csv")))
cov_data_fit = DataFrame(load(projectdir("figure_rates", "data", "cov_data_fit.csv")))

t_cov_lit, cov_lit, var_cov_lit = eachcol(cov_data_lit[:, [:t, :Cov, :Var]])
Δcov_lit = sqrt.(var_cov_lit) ./ sqrt(length(var_cov_lit));
t_cov_fit, cov_fit, var_cov_fit = eachcol(cov_data_fit[:, [:t, :Cov, :Var]])
Δcov_fit = sqrt.(var_cov_fit) ./ sqrt(length(var_cov_fit));

band!(axb, t_cov_lit, cov_lit - Δcov_lit, cov_lit + Δcov_lit, color=(color_lit, 0.5))
lines!(axb, t_cov_lit, cov_lit, color=color_lit)

band!(axb, t_cov_fit, cov_fit - Δcov_fit, cov_fit + Δcov_fit, color=(color_fit, 0.5))
lines!(axb, t_cov_fit, cov_fit, color=color_fit)

t = range(t_cov_lit[begin], t_cov_lit[end], length=100)
mattingly_cov(t; σ, τ) = @. σ^2 * exp(-t / τ)
band!(axb, t, mattingly_cov(t, σ=0.092 + 0.002, τ=11.75 + 0.04), mattingly_cov(t, σ=0.092 - 0.002, τ=11.75 - 0.04), color=(color_mattingly, 0.5))
lines!(axb, t, mattingly_cov(t, σ=0.092, τ=11.75), color=color_mattingly)
Makie.xlims!(axb, -0.3, 20)
Makie.ylims!(axb, -5e-4, nothing)


axc = Axis(fig[3, 1:2], xlabel = "gradient steepness (mm⁻¹)", title = "information rate (bits/s)")

gaussian_rates = DataFrame(load(projectdir("figure_rates", "data", "gaussian_rates.csv")))

color_map = Dict(
    "Mattingly" => color_mattingly,
    "Literature" => color_lit,
    "Fit" => color_fit
)

for row in eachrow(gaussian_rates)
    β = row.β
    c = color_map[row.label]
    g = 0:1e-2:0.45
    lines!(axc, g, g .^ 2 .* β, color=c)
end


pws_data = DataFrame(load(projectdir("figure_rates", "data", "2023-02-17_pws_info_rates_lit.csv")))
pws_extra_data = DataFrame(load(projectdir("figure_rates", "data", "2023-01-15_pws_extrapolation.csv")))

canonical_pws = pws_data[pws_data.NClusters .== 400 .&& pws_data.Lmax .== 6, :]
fitted_pws = pws_extra_data[pws_extra_data.Lmax .== 15, :]

scatter!(axc, canonical_pws.g .* 1e3, canonical_pws.info_rate; color=color_lit)
scatter!(axc, fitted_pws.g .* 1e3, fitted_pws.extrapolation; color=color_fit)

xlims!(axc, 0, 0.42)
ylims!(axc, 0, 0.17)

group_color = map(colors) do c
    PolyElement(color = c, strokecolor = :transparent)
end

group_method = [
    MarkerElement(marker = :circle, color = :black),
    LineElement(linestyle = :solid, color = :black)
]

Legend(
    fig[2, 1:2],
    [group_color],
    [labels],
    [""];
    orientation=:horizontal,
    titlehalign=:center,
)

axislegend(
    axc,
    [group_method],
    [["PWS", "Gaussian approx."]],
    ["Method"];
    position=:lt
)

# Tweaks to layout

rowsize!(fig.layout, 1, Auto(0.66))

save(projectdir("figure_rates", "chemotaxis_comparison.pdf"), fig, pt_per_unit = 1)
save(projectdir("figure_rates", "chemotaxis_comparison.png"), fig, pt_per_unit = 1)


fig