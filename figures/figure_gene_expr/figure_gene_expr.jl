
using CSVFiles
using DataFrames
using DrWatson

ddir = projectdir("figures", "figure_gene_expr", "data")

comb_data = DataFrame(load(joinpath(ddir, "gene-expression_2021-10-04.csv")))
dmc_data = comb_data[comb_data.Alg.=="Direct MC", :]
smc_data = comb_data[comb_data.Alg.=="SMC", :]
smc_estimate = groupby(smc_data, :M)[(; M = 128)]
ti_data = comb_data[(comb_data.Alg.=="TI").&(comb_data.Duration.==comb_data.DiscreteTimes), :]
zechner_data = DataFrame(load(joinpath(ddir, "gene-expression_2021-06-18_zechner.csv")))
gaussian_data = DataFrame(load(joinpath(ddir, "gene-expression_2021-07-19_gaussian.csv")))


using CairoMakie

presentation_theme = Theme(font = "Noto Sans Regular", linewidth=3)

fig1 = with_theme(presentation_theme) do
    f = Figure(resolution=(600, 600))
    ax = Axis(f[1, 1], xlabel="trajectory duration", ylabel="mutual information (nats)")
    dmc_d = groupby(dmc_data, :M)[end]
    # dmc_lines = lines!(ax, dmc_d.DiscreteTimes, dmc_d.Mean, label="direct")
    smc_lines = lines!(ax, smc_estimate.DiscreteTimes, smc_estimate.Mean, color="black")
    # zec_lines = lines!(zechner_data.Duration, zechner_data.PMI, linewidth=2, linestyle=:dash, color=:red)
    # lines!(gaussian_data.DiscreteTimes, gaussian_data.Value, linewidth = 4, linestyle = :dot, color = "#2b8cbe")


    # legend = Legend(
    #     f[2, 1], 
    #     [dmc_lines, smc_lines, zec_lines],
    #     ["direct", "segment-by-segment", "Duso, Zechner (2019)"],
    #     orientation = :horizontal,
    #     tellheight = true
    #     # [group_color, ti_color, smc_color, approx_entry], 
    #     # [m_label, ["256 MCMC steps", "4096 MCMC steps"], 
    #     # ["M = 128"], ["Duso, Zechner (2019)", "Gaussian"]], 
    #     # ["brute force", "thermodynamic\nintegration", "particle filter", "approximations"], 
    #     # rowgap=0, 
    #     # titlegap=2, 
    #     # groupgap=5,
    #     # patchsize=(10,10),
    #     # padding=(3,3,3,3),
    #     # linewidth=1.0,
    #     # gridshalign=:left,
    #     # titlehalign=:left,
    #     # titlefont="Noto Sans Medium",
    #     # framevisible=false
    # )
    xlims!(ax, 0, 8)
    ylims!(ax, 0, 16)

    f
end
save(projectdir("figures", "figure_gene_expr", "gene-expr-figure-p.png"), fig1, pt_per_unit = 1)

normalfont = projectdir("figures", "fonts", "NotoSans-Condensed.ttf")
titlefont = projectdir("figures", "fonts", "NotoSans-SemiBoldItalic.ttf")

fontsize_theme = Theme(
    font = normalfont,
    fontsize = 8,
    colgap = 3,
    linewidth = 2,
    markersize = 6,
    spinewidth = 0.5,
    Axis = (
        spinewidth = 1.0,
        xgridwidth = 1.0,
        ygridwidth = 1.0,
        xtickwidth = 1.0,
        ytickwidth = 1.0,
        xticksize = 3,
        yticksize = 3,
        xlabelpadding = 0,
        ylabelpadding = 5,
        xticklabelpad = 0,
        yticklabelpad = 3,
    ),
    Legend = (
        framewidth = 1.0,
        rowgap = 0,
        titlegap = 2,
        groupgap = 5,
        patchsize = (11, 10),
        padding = (3, 3, 3, 3),
        linewidth = 1.25,
        gridshalign = :left,
        titlehalign = :left,
        titlefont = titlefont,
        framevisible = false
    )
)

fig = with_theme(fontsize_theme) do
    f = Figure(resolution = (246, 1.4 * 146), figure_padding = 2)
    Axis(f[1, 1], xlabel = "trajectory duration", ylabel = "mutual information (nats)")

    groups = collect(groupby(dmc_data, :M; sort = true))

    m_label = ["M = $(g.M[1])" for g in groups]
    cmap = cgrad(:linear_kgy_5_95_c69_n256, rev = true)
    colors = [cmap[i/length(groups)*0.7] for i = 1:length(groups)]
    ti_colors = [cgrad(:lajolla)[i] for i = range(0.2, 0.6, length = 3)]
    pathmi_color = :orangered

    for (i, group) in enumerate(groups)
        label = "$(group.M[1])"
        lines!(group.DiscreteTimes, group.Mean, linewidth = 0.75, label = label, color = colors[i])
    end

    group_color = [LineElement(color = color) for color in colors]
    smc_color = [LineElement(color = :black)]

    lines!(smc_estimate.DiscreteTimes, smc_estimate.Mean, label = "SMC", linewidth = 1.7, color = :black)
    lines!(zechner_data.Duration, zechner_data.PMI, linewidth = 1.5, linestyle = :dash, color = pathmi_color)
    lines!(gaussian_data.DiscreteTimes, gaussian_data.Value, linewidth = 1.5, linestyle = :dot, color = "#2b8cbe")

    ti_label = []
    ti_color = []
    for (i, group) in enumerate(groupby(ti_data, :M, sort = true))
        scatter!(group.DiscreteTimes, group.Mean, marker = :cross, color = ti_colors[i])
        push!(ti_label, "M = $(group.M[1])")
        push!(ti_color, MarkerElement(marker = :cross, color = ti_colors[i]))
    end

    approx_entry = [LineElement(linestyle = :dash, color = pathmi_color), LineElement(linestyle = :dot, color = "#2b8cbe")]

    legend = Legend(
        f[1, 2],
        [group_color, ti_color, smc_color, approx_entry],
        [m_label, ti_label,
            ["M = 128"], ["Duso, et. al. (2019)", "Gaussian (LNA)"]],
        ["DPWS", "TI-PWS", "RR-PWS", "approximations"],
    )
    xlims!(0.0, 10.2)
    ylims!(0.0, 42)

    inset_lims = (3.3, 8.5)
    lines!([(0, 0), (inset_lims[1], 0), inset_lims, (0, inset_lims[2]), (0, 0)], color = :black, linewidth = 0.75, linestyle = :dot)

    inset_ax = Axis(f[1, 1],
        width = Relative(0.55),
        height = Relative(0.33),
        halign = 0.25,
        valign = 0.95,
        xtickalign = 1.0,
        ytickalign = 1.0,
        backgroundcolor = :white,
        xticksize = 2,
        xticklabelsize = 7,
        xgridvisible = true,
        yticksize = 2,
        yticklabelsize = 7,
    )

    for (i, group) in enumerate(groups)
        label = "$(group.M[1])"
        lines!(inset_ax, group.DiscreteTimes, group.Mean, linewidth = 0.5, label = label, color = colors[i])
    end
    lines!(inset_ax, smc_estimate.DiscreteTimes, smc_estimate.Mean, linewidth = 1.0, color = :black)
    lines!(inset_ax, zechner_data.Duration, zechner_data.PMI, linewidth = 0.75, linestyle = :dash, color = pathmi_color)
    lines!(inset_ax, gaussian_data.DiscreteTimes, gaussian_data.Value, linewidth = 1.5, linestyle = :dot, color = "#2b8cbe")

    for (i, group) in enumerate(groupby(ti_data, :M, sort = true))
        scatter!(inset_ax, group.DiscreteTimes, group.Mean, marker = :cross, color = ti_colors[i])
    end

    xlims!(inset_ax, 0, inset_lims[1])
    ylims!(inset_ax, 0, inset_lims[2])
    translate!(inset_ax.scene, 0, 0, 10)
    translate!(inset_ax.elements[:background], 0, 0, 9)
    translate!(inset_ax.elements[:xgridlines], 0, 0, 9)
    translate!(inset_ax.elements[:ygridlines], 0, 0, 9)

    f
end

save(projectdir("figures", "figure_gene_expr", "gene-expr-figure.svg"), fig, pt_per_unit = 1)
save(projectdir("figures", "figure_gene_expr", "gene-expr-figure.pdf"), fig, pt_per_unit = 1)

fig
