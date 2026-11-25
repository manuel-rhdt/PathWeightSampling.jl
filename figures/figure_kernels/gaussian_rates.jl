#!/usr/bin/env julia
#
# Compute the Gaussian information rates of the comparison figure (`figure_rates/figure_comparison.jl`)
# from the noise autocovariance and the response kernel of the chemotaxis model.
#
# In the Gaussian approximation the information rate is R = β g² for small gradient steepness g, with
#
#     β = 1/(4π ln 2) ∫ dω V(ω) |K(ω)|² / N(ω)
#
# where V is the power spectrum of the input signal, K the Fourier transform of the response kernel and N
# the power spectrum of the noise. The spectra are computed from `cov_data_*.csv` and `kern_data_*.csv`
# (see `simulate_kernels.jl`); for the experiments of Mattingly et al. they are given analytically.
#
# Run it from the repository root with the *figures* environment:
#
#     julia --project=figures figures/figure_kernels/gaussian_rates.jl
#     julia --project=figures figures/figure_kernels/gaussian_rates.jl --data figures/figure_kernels/data/reproduced
#
# Options
#   --data   directory with cov_data_{lit,fit}.csv and kern_data_{lit,fit}.csv (default: the committed data)
#   --out    directory for the result `gaussian_rates.csv` (default: next to the data in `reproduced/`)
#   --tolerance  exit with an error if a β differs from the committed value by more than this relative amount
#                (for tests: with the committed data and the default window the difference is zero)
#   --window lanczos | hanning: window applied to the time series before the Fourier transform
#            (default lanczos, which reproduces the committed gaussian_rates.csv; figure_kernels.jl uses
#            hanning for the plotted spectra, which changes β by about 0.2 %)
#
# The result is compared with the committed figure_rates/data/gaussian_rates.csv.

using DataFrames
using CSVFiles
using DSP
using FFTW
using Printf

const HERE = @__DIR__

# input signal and the analytic spectra of Mattingly et al.
const aᵥ = 157.1
const λ = 0.862
const G = 1.73
const τ₁ = 0.017
const τ₂ = 9.9
const λ_m = 1 / 11.75
const D_n = 7.2e-4

vel(ω) = 2 * aᵥ * λ / (ω^2 + λ^2) ./ 1000^2
ker(ω) = G^2 / τ₁^2 / ((1 / τ₂^2 + ω^2) * ((1 / τ₁ + 1 / τ₂)^2 + ω^2))
noi(ω) = 2 * D_n / (ω^2 + λ_m^2)

function mattingly_kernels()
    ω = 0:1e-2:100
    DataFrame(freq=ω, input=vel.(ω), response=ker.(ω), noise=noi.(ω))
end

"""Power spectra of input, response and noise from the response kernel and the noise autocovariance."""
function get_kernels(kern_data, cov_data; window=hanning)
    t = kern_data.t
    sr = 1 / (t[2] - t[1])
    n = length(kern_data.t)
    t_n = inv(sr) * (-2(n-1):2(n-1))
    input_cov = @. aᵥ * exp(-λ * abs(t_n)) / (1000^2) # divide by 1000^2 for unit conversion

    extend_end(x) = vcat(x, fill(x[end], length(x) - 1))

    kern = extend_end(kern_data.Kernel)
    kernel_window = window(2 * length(kern) - 1)
    kernel = kernel_window .* vcat(zeros(length(kern) - 1), kern)

    cov = extend_end(cov_data.Cov)
    noise_window = window(2 * length(cov) - 1)
    noise = noise_window .* vcat(cov[end:-1:begin+1], cov)

    # Take the fourier transform and divide by the sample rate to get the fourier kernels
    input_kernel = abs.(rfft(fftshift(input_cov))) ./ sr
    kernel_squared = abs2.(rfft(fftshift(kernel))) ./ sr^2
    noise_kernel = abs.(rfft(fftshift(noise))) ./ sr

    freq = rfftfreq(length(kernel), sr) .* (2π)

    DataFrame(freq=freq, input=input_kernel, response=kernel_squared, noise=noise_kernel)
end

function info_rate_coef(kernels)
    ω = kernels.freq
    dω = ω[2] - ω[1]
    dω * sum(kernels.input .* kernels.response ./ kernels.noise) / (log(2) * 2π)
end

function parse_options(args)
    opts = Dict("data" => joinpath(HERE, "data"), "out" => nothing, "window" => "lanczos", "tolerance" => nothing)
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("unexpected argument $arg")
        name, value = occursin('=', arg) ? split(arg[3:end], '=', limit=2) : (arg[3:end], nothing)
        if value === nothing
            i += 1
            i <= length(args) || error("missing value for $arg")
            value = args[i]
        end
        haskey(opts, name) || error("unknown option --$name")
        opts[name] = value
        i += 1
    end
    opts["out"] === nothing && (opts["out"] = joinpath(opts["data"], "reproduced"))
    opts["window"] in ("hanning", "lanczos") || error("--window must be hanning or lanczos")
    opts
end

function main(args)
    o = parse_options(args)
    window = o["window"] == "hanning" ? hanning : lanczos
    read(name) = DataFrame(load(joinpath(o["data"], name)))
    kernels = Dict(
        "Mattingly" => mattingly_kernels(),
        "Literature" => get_kernels(read("kern_data_lit.csv"), read("cov_data_lit.csv"); window),
        "Fit" => get_kernels(read("kern_data_fit.csv"), read("cov_data_fit.csv"); window),
    )
    labels = ["Mattingly", "Literature", "Fit"]
    colors = ["green", "red", "blue"]
    betas = [info_rate_coef(kernels[l]) for l in labels]

    mkpath(o["out"])
    outfile = joinpath(o["out"], "gaussian_rates.csv")
    open(outfile, "w") do io
        println(io, "\"β\",\"plot_color\",\"label\"")
        for (β, c, l) in zip(betas, colors, labels)
            @printf(io, "%.17g,%s,\"%s\"\n", β, c, l)
        end
    end
    println("wrote ", outfile)

    published = DataFrame(load(joinpath(HERE, "..", "figure_rates", "data", "gaussian_rates.csv")))
    println("\nβ (bits/s per (mm⁻¹)²), window = ", o["window"])
    println(rpad("model", 12), rpad("new", 24), rpad("committed", 24), "relative difference")
    worst = 0.0
    for (l, β) in zip(labels, betas)
        pub = published.β[findfirst(==(l), published.label)]
        worst = max(worst, abs((β - pub) / pub))
        println(rpad(l, 12), rpad(@sprintf("%.10f", β), 24), rpad(@sprintf("%.10f", pub), 24), @sprintf("%+.2e", (β - pub) / pub))
    end
    if o["tolerance"] !== nothing && worst > parse(Float64, o["tolerance"])
        println(stderr, "β differs from the committed values by ", worst, " (tolerance ", o["tolerance"], ")")
        exit(1)
    end
end

main(ARGS)
