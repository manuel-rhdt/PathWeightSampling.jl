
using CSVFiles
using DataFrames
using DrWatson

ddir = projectdir("plots", "figure_gene_expr", "data")

dmc_data = DataFrame(load(joinpath(ddir, "gene-expression_2021-06-17_directmc.csv")))
smc_data = DataFrame(load(joinpath(ddir, "gene-expression_2021-06-21_smc.csv")))
ti_data  = DataFrame(load(joinpath(ddir, "gene-expression_2021-06-17_thermodynamic-integration.csv")))
ti_data2  = DataFrame(load(joinpath(ddir, "gene-expression_2021-06-18_thermodynamic-integration.csv")))
zechner_data = DataFrame(load(joinpath(ddir, "gene-expression_2021-06-18_zechner.csv")))
gaussian_data = DataFrame(load(joinpath(ddir, "gene-expression_2021-07-19_gaussian.csv")))

smc_estimate = groupby(smc_data, :M)[(;M=128)]
ti_estimates = ti_data[ti_data.Duration .== ti_data.DiscreteTimes, :]
ti_estimates2 = ti_data2[ti_data.Duration .== ti_data.DiscreteTimes, :]


using CairoMakie

fontsize_theme = Theme(
    font = "Noto Sans Regular", 
    fontsize = 8, 
    colgap=3, 
    linewidth=2, 
    markersize=5,
    spinewidth=0.5,
    Axis = (
        spinewidth=0.5,
        xgridwidth=0.5,
        ygridwidth=0.5,
        xtickwidth=0.5,
        ytickwidth=0.5,
        xticksize=3, 
        yticksize=3,
        xlabelpadding=0,
        ylabelpadding=0,
        xticklabelpad=0,
        yticklabelpad=3,
    ),
    Legend = (
        framewidth=0.5,
    )
)
fig = with_theme(fontsize_theme) do 
    f = Figure(resolution=(246, 1.4*146), figure_padding = 2)
    Axis(f[1, 1], xlabel="trajectory duration", ylabel="mutual information (nats)")
    groups = collect(groupby(dmc_data, :M)[begin+2:3:end])
    m_label = ["M = $(g.M[1])" for g in groups]
    cmap = cgrad(:linear_kgy_5_95_c69_n256, rev=true)
    colors = [cmap[i / length(groups) * 0.7] for i = 1:length(groups)]
    ti_colors = [:BurlyWood, :saddlebrown]

    for (i, group) in enumerate(groups)
        label = "$(group.M[1])"
        lines!(group.DiscreteTimes, group.Mean, linewidth=0.75, label=label, color=colors[i])
    end

    group_color = [LineElement(color=color) for color in colors]
    smc_color = [LineElement(color=:black)]

    ti_color = [MarkerElement(marker=:cross, color=ti_colors[1]), MarkerElement(marker=:cross, color=ti_colors[2])]

    lines!(smc_estimate.DiscreteTimes, smc_estimate.Mean, label="SMC", linewidth=1.5, color=:black)
    lines!(zechner_data.Duration, zechner_data.PMI, linewidth=1.5, linestyle=:dash, color=:red)
    lines!(gaussian_data.DiscreteTimes, gaussian_data.Value, linewidth=1.5, linestyle=:dot, color=:MidnightBlue)

    ti_points = scatter!(ti_estimates.DiscreteTimes, ti_estimates.Mean, marker=:cross, label="Thermodynamic Integration", color=ti_colors[1])
    ti_points2 = scatter!(ti_estimates2.DiscreteTimes, ti_estimates2.Mean, marker=:cross, label="Thermodynamic Integration", color=ti_colors[2])

    approx_entry = [LineElement(linestyle=:dash, color=:red), LineElement(linestyle=:dot, color=:MidnightBlue)]

    legend = Legend(
        f[1, 2], 
        [group_color, ti_color, smc_color, approx_entry], 
        [m_label, ["256 MCMC steps", "4096 MCMC steps"], 
        ["M = 128"], ["Duso, et.al. (2019)", "Gaussian"]], 
        ["brute force", "thermodynamic\nintegration", "particle filter", "approximations"], 
        rowgap=0, 
        titlegap=2, 
        groupgap=5,
        patchsize=(10,10),
        padding=(3,3,3,3),
        linewidth=1.0,
        gridshalign=:left,
        titlehalign=:left,
        titlefont="Noto Sans Medium",
        framevisible=false
    )
    ylims!(-2.5, 115)

    inset_lims = (3, 8)
    lines!([(0,0), (inset_lims[1], 0), inset_lims, (0, inset_lims[2]), (0,0)], color=:black, linewidth=0.5, linestyle=:dash)

    inset_ax = Axis(f[1, 1],
        width=Relative(0.5),
        height=Relative(0.3),
        halign=0.25,
        valign=0.95,
        backgroundcolor=:white,
        xticksize=3,
        xticklabelsize=7,
        yticksize=2,
        yticklabelsize=7,
    )

    for (i, group) in enumerate(groups)
        label = "$(group.M[1])"
        lines!(inset_ax, group.DiscreteTimes, group.Mean, linewidth=0.5, label=label, color=colors[i])
    end
    lines!(inset_ax, smc_estimate.DiscreteTimes, smc_estimate.Mean, linewidth=0.75, color=:black)
    lines!(inset_ax, zechner_data.Duration, zechner_data.PMI, linewidth=0.75, linestyle=:dash, color=:red)
    lines!(inset_ax, gaussian_data.DiscreteTimes, gaussian_data.Value, linewidth=1.5, linestyle=:dot, color=:MidnightBlue)

    ti_points = scatter!(inset_ax, ti_estimates.DiscreteTimes, ti_estimates.Mean, marker=:cross, label="Thermodynamic Integration", color=ti_colors[1])
    ti_points2 = scatter!(inset_ax, ti_estimates2.DiscreteTimes, ti_estimates2.Mean, marker=:cross, label="Thermodynamic Integration", color=ti_colors[2])
    xlims!(inset_ax, 0, inset_lims[1])
    ylims!(inset_ax, 0, inset_lims[2])
    translate!(inset_ax.scene, 0, 0, 10)
    translate!(inset_ax.elements[:background], 0, 0, 9)

    f
end

save(projectdir("plots", "figure_gene_expr", "gene-expr-figure.pdf"), fig, pt_per_unit = 1)
