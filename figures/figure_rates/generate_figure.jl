using DrWatson
using DataFrames
using Statistics
using CairoMakie
using LaTeXStrings
using CSVFiles

normalfont = projectdir("figures", "fonts", "NotoSans-Condensed.ttf")
italicfont = projectdir("figures", "fonts", "NotoSans-Italic.ttf")
boldfont = projectdir("figures", "fonts", "NotoSans-CondensedSemiBold.ttf")
subfigurelabelfont = projectdir("figures", "fonts", "NotoSans-Bold.ttf")

# plot_theme = Theme(
#     font = normalfont,
#     fontsize = 8,
#     colgap = 4,
#     rowgap = 0,
#     linewidth = 2,
#     markersize = 5,
#     spinewidth = 0.5,
#     figure_padding = (2, 2, 2, 2),
#     Axis = (
#         spinewidth = 1.0,
#         xgridwidth = 1.0,
#         xminorgridwidth = 1.0,
#         yminorgridwidth = 1.0,
#         ygridwidth = 1.0,
#         xtickwidth = 1.0,
#         ytickwidth = 1.0,
#         xticksize = 3,
#         yticksize = 3,
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

plot_theme = Theme(
    font = normalfont,
    fontsize = 48,
    colgap = 4,
    rowgap = 0,
    linewidth = 5,
    markersize = 12.5,
    figure_padding = (2, 2, 2, 2),
    Axis = (
        spinewidth = 3.0,
        xgridwidth = 3.0,
        xminorgridwidth = 2.0,
        yminorgridwidth = 2.0,
        ygridwidth = 3.0,
        xtickwidth = 3.0,
        ytickwidth = 3.0,
        xticksize = 10,
        yticksize = 10,
        xlabelpadding = 0,
        ylabelpadding = 2,
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

# fig = Figure(resolution = (246, 240))
fig = Figure(resolution = (1280, 800))
ax1 = Axis(fig[1, 1], xlabel = L"gradient steepness ($\mathrm{mm}^{-1}$)", ylabel = "information rate (bits/s)")

canonical_gaussian = DataFrame(load(projectdir("figures", "figure_rates", "data", "canonical_gaussian.csv")))
canonical_gaussian400 = DataFrame(load(projectdir("figures", "figure_rates", "data", "canonical_gaussian_Nc=400.csv")))

canonical_pws = DataFrame(load(projectdir("figures", "figure_rates", "data", "canonical_pws.csv")))
fitted_gaussian = DataFrame(load(projectdir("figures", "figure_rates", "data", "fitted_gaussian.csv")))


lines!(ax1, canonical_gaussian.Gradient .* 1e3, canonical_gaussian.InfoRate; label="Canonical Model (800 clusters)\nGaussian")
scatter!(ax1, canonical_pws.g .* 1e3, canonical_pws.info_rate; label="Canonical Model (800 clusters)\nPWS")

# lines!(ax1, canonical_gaussian400.Gradient .* 1e3, canonical_gaussian400.InfoRate; label="Canonical Model (400 clusters)\nGaussian")


# lines!(ax1, fitted_gaussian.Gradient .* 1e3, fitted_gaussian.InfoRate; label="Fitted Model (10 clusters)\nGaussian")

g_grid = range(0, 0.4e-3, length=51)
# lines!(ax1, g_grid .* 1e3, 0.22e6 .* (g_grid .^ 2), label="Experiments\nMattingly et al.")

# leg = Legend(fig[1, 2], ax1, orientation = :vertical, tellwidth = true)

save(projectdir("figures", "figure_rates", "rates.pdf"), fig, pt_per_unit = 1)
save(projectdir("figures", "figure_rates", "rates.png"), fig, pt_per_unit = 1)


fig