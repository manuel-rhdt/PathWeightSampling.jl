using GaussianMcmc
using Plots

system = GaussianMcmc.chemotaxis_system(
    mean_L = 20,
    num_receptors = 10_000,
    Y_tot = 5_000,
    LR_timescale = 1e-2,
    Y_timescale = 1e-1,
    Y_ratio = 1/8,
    LR_ratio = 0.5,
    q=0,
    dtimes = 0:0.04:2
)

@time sol = GaussianMcmc._solve(system)

p = plot(sol)

@time conf = GaussianMcmc.generate_configuration(system)

cond_ens = GaussianMcmc.ConditionalEnsemble(system)

alg = GaussianMcmc.SMCEstimate(256)

@profview result = GaussianMcmc.simulate(alg, conf, cond_ens).samples

result

using Distributions

for x in eachrow(result)
    @show mean(x) / var(x)
end

θ = 11.5
p = plot(xlabel="Path Action", ylabel="Probability")
fitted_dists = []
for (c, i) in enumerate(11:10:51)
    x = result[i,:]
    t = dtimes[i]
    x = (mean(x) + var(x) / θ) .- x
    plot!(p, x, seriestype=:stephist, xlim=(-10,400), fill=true, normalize=true, fillalpha=0.1, label="T=$t", color=c)

    fitted_gamma = @show fit(Gamma{Float64}, x)
    push!(fitted_dists, fitted_gamma)
    plot!(p, 0:1:400, y->pdf(fitted_gamma, y), color=c, label="Gamma distribution fit")
end
p
savefig(plot(p, dpi=300), "~/Downloads/histograms.png")

p = plot(result[end,:], seriestype=:stephist, fill=true, legend=false, yscale=:log10, ylim=(1e-6,1e-2), normalize=true)
savefig(p, "~/Downloads/histogram.png")

using Statistics

function bootstrap(f, data::AbstractVector; bsamples=100)
    N = length(data)
    map(1:bsamples) do i
        sample_indices = rand(1:N, N)
        f(@view data[sample_indices])
    end
end

GaussianMcmc.logmeanexp(result[end, :])

bdata = bootstrap(GaussianMcmc.logmeanexp, result[end,:], bsamples=10^4)
h = histogram(bdata, title="Bootstrapping Histogram", xlabel="Path Action", ylabel="Count")
savefig(h, "~/Downloads/bootstrap.png")

using Distributions 

p=plot()
for (i, shape_param) in enumerate([2, 5, 10, 20])
    dist = Gamma(shape_param, 1.0)
    plot!(p, 0.0:0.01:30, x->pdf(dist, x), color=i, linestyle=:solid, label="Gamma α=$shape_param")
    plot!(p, 0.0:0.01:30, x->pdf(dist, x)*exp(-x)*(scale(dist) + 1)^shape(dist), color=i, linestyle=:dot, fill=true, fillalpha=0.1, label="exp(-x)*Gamma(x)")
end
p
savefig(plot(p, dpi=300), "~/Downloads/gamma.png")


p=plot()
for (i, dist) in enumerate(fitted_dists)
    plot!(p, 0.0:1:400, x->pdf(dist, x), color=i, linestyle=:solid)
    plot!(p, 0.0:1:400, x->pdf(dist, x)*exp(-x)*(scale(dist) + 1)^shape(dist), color=i, linestyle=:dot, fill=true, fillalpha=0.1)
end
p
