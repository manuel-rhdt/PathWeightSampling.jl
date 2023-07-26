using DrWatson
using DataFrames
using Statistics
using CairoMakie
using LaTeXStrings
using CSVFiles
using DSP
using FFTW
using Interpolations

normalfont = projectdir("fonts", "NotoSans-Condensed.ttf")
italicfont = projectdir("fonts", "NotoSans-Italic.ttf")
boldfont = projectdir("fonts", "NotoSans-CondensedSemiBold.ttf")
subfigurelabelfont = projectdir("fonts", "NotoSans-Bold.ttf")

cov_data_lit = DataFrame(load(projectdir("figure_kernels", "data", "cov_data_lit.csv")))
cov_data_fit = DataFrame(load(projectdir("figure_kernels", "data", "cov_data_fit.csv")))

kern_data_lit = DataFrame(load(projectdir("figure_kernels", "data", "kern_data_lit.csv")))
kern_data_fit = DataFrame(load(projectdir("figure_kernels", "data", "kern_data_fit.csv")))

# Mattingly
aᵥ = 157.1
λ = 0.862
G = 1.73
τ₁ = 0.017
τ₂ = 9.9
σ_y = 0.01
σ_m = 0.092
λ_y = 1/0.1
λ_m = 1/11.75
D_n = 7.2e-4

vel(ω) = 2*aᵥ*λ / (ω^2 + λ^2) ./ 1000^2
ker(ω) = G^2 / τ₁^2 / ((1/τ₂^2 + ω^2) * ( (1/τ₁ + 1/τ₂)^2 + ω^2))
noi(ω) = 2 * D_n / (ω^2 + λ_m^2)
ω = 0:1e-2:100
mattingly_kernels = DataFrame(freq=ω, input=vel.(ω), response=ker.(ω), noise=noi.(ω))


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
        valign = :center,
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

set_theme!(plot_theme)

begin

fig = Figure(resolution = (510, 250))

function get_kernels(kern_data, cov_data)
    t = kern_data.t
    sr = 1/(t[2] - t[1])
    n = length(kern_data.t)
    t_n = inv(sr) * (-2(n-1):2(n-1))
    input_cov = @. aᵥ * exp(-λ * abs(t_n)) / (1000^2) # divide by 1000^2 for unit conversion
    
    extend_end(x) = vcat(x, fill(x[end], length(x)-1))
    
    kern = extend_end(kern_data.Kernel)
    kernel_window = hanning(2*length(kern) - 1)
    kernel = kernel_window .* vcat(zeros(length(kern)-1), kern)
    
    cov = extend_end(cov_data.Cov)
    noise_window = hanning(2*length(cov) - 1)
    noise = noise_window .* vcat(cov[end:-1:begin+1], cov)
    
    # Take the fourier transform and divide by the sample rate to get the fourier kernels
    input_kernel = abs.(rfft(fftshift(input_cov))) ./ sr
    kernel_squared = abs2.(rfft(fftshift(kernel))) ./ sr^2
    noise_kernel = abs.(rfft(fftshift(noise))) ./ sr
    
    freq = rfftfreq(length(kernel), sr) .* (2π)
    
    DataFrame(freq=freq, input=input_kernel, response=kernel_squared, noise=noise_kernel)
end

function plot_freq_panels!(axs, kernels; kwargs...)
    freq = 10 .^ range(-2,2,length=100)
    input_kernel = LinearInterpolation(kernels.freq, kernels.input)(freq)
    kernel_squared = LinearInterpolation(kernels.freq, kernels.response)(freq)
    noise_kernel = LinearInterpolation(kernels.freq, kernels.noise)(freq)

    dash = [0, 5.0, 8.0]

    lines!(ax1, freq, kernel_squared, linestyle=:dot; kwargs...)
    lines!(ax1, freq, noise_kernel, linestyle=dash; kwargs...)
    Makie.xlims!(ax1, 1e-2, 1e2)
    Makie.ylims!(ax1, 1e-10, 1e5)

    lines!(ax2, freq, kernel_squared ./ kernel_squared[begin], linestyle=:dot, label=L"|K(\omega)|^2"; kwargs...)
    lines!(ax2, freq, noise_kernel ./ noise_kernel[begin], linestyle=dash, label=L"N(\omega)"; kwargs...)
    Makie.xlims!(ax2, 0, 5.0)
    Makie.ylims!(ax2, 0, 1.1)

    lines!(ax3, freq, freq .* input_kernel .* kernel_squared ./ noise_kernel ./ log(2); kwargs...)
    Makie.xlims!(ax3, 1e-2, 1e2)

    lines!(ax4, freq, input_kernel .* kernel_squared ./ noise_kernel ./ log(2); kwargs...)
    Makie.xlims!(ax4, 0, 5.0)
    
    nothing
end

palette = cgrad(:Egypt, categorical=true)
# palette = cgrad(:glasbey_bw_minc_20_maxl_70_n256, categorical=true)

color_mattingly = palette[1]
color_fit = palette[2]
color_lit = palette[3]

model_colors = [color_mattingly, color_lit, color_fit]
model_group = [PolyElement(color = c) for c in model_colors]
model_label = ["Experiments", "Literature", "Fitted"]

kernel_style = [:solid, :dot, :dash]
kernel_group = [LineElement(linestyle=ls, color=:black) for ls in kernel_style]
kernel_label = [L"V(\omega)", L"|K(\omega)|^2", L"N(\omega)"]

ax1 = Axis(fig[1,1], xscale=log10, yscale=log10, ylabel=L"F(\omega)")
ax2 = Axis(fig[1,2], ylabel=L"F(\omega) / F(0)")
ax3 = Axis(fig[2,1], xscale=log10, xlabel=L"Frequency $\omega$ (rad/s)", ylabel=L"\omega\;V |K|^2 / N")
ax4 = Axis(fig[2,2], xlabel=L"Frequency $\omega$ (rad/s)", ylabel=L"V |K|^2 / N")

lines!(ax1, mattingly_kernels.freq, mattingly_kernels.input, color=:black)
lines!(ax2, mattingly_kernels.freq, mattingly_kernels.input ./ mattingly_kernels.input[begin], label=L"V(\omega)", color=:black)

lit_kernels = get_kernels(kern_data_lit, cov_data_lit)
fit_kernels = get_kernels(kern_data_fit, cov_data_fit)

plot_freq_panels!([ax1, ax2, ax3, ax4], mattingly_kernels, color=color_mattingly)
plot_freq_panels!([ax1, ax2, ax3, ax4], lit_kernels, color=color_lit)
plot_freq_panels!([ax1, ax2, ax3, ax4], fit_kernels, color=color_fit)
axislegend(ax2, [kernel_group], [kernel_label], ["Kernel"])

Legend(fig[1:2,3], [model_group], [model_label], [""])

save(projectdir("figure_kernels", "fourier_comparison.pdf"), fig, pt_per_unit = 1)
save(projectdir("figure_kernels", "fourier_comparison.png"), fig, pt_per_unit = 1)

fig

end