include("basic_setup.jl")

using DataFrames

function prepare(mean_s)
    t_s = 100
    t_x = 20

    κ = mean_s / t_s
    λ = 1 / t_s
    ρ = 1 / t_x
    μ = 1 / t_x
    mean_x = mean_s * ρ / μ

    # see Tostevin, ten Wolde, eq. 27
    sigma_squared_ss = mean_s
    sigma_squared_sx = ρ * mean_s / (λ + μ)
    sigma_squared_xx = mean_x * (1 + ρ / (λ + μ))

    joint_stationary = MvNormal([mean_s, mean_x], [sigma_squared_ss sigma_squared_sx; sigma_squared_sx sigma_squared_xx])
    signal_stationary = Poisson(mean_s)

    gen = Trajectories.configuration_generator(sn, rn, [κ, λ], [ρ, μ], signal_stationary, joint_stationary)

    (system, initial) = Trajectories.generate_configuration(gen; duration=100.0)
end

function get_results(algorithm, num_evals, system, initial)
    results = zeros(num_evals)
    Threads.@threads for i in 1:num_evals
        r = Trajectories.simulate(algorithm, initial, system)
        results[i] = Trajectories.log_marginal(r)
    end
    DataFrame(log_marginal=results, κ=system.sparams[1], λ=system.sparams[2], estimate=Trajectories.name(algorithm))
end


an = AnnealingEstimate(15, 50, 100)
ti = TIEstimate(1024, 6, 2^14)
direct = DirectMCEstimate(50_000)

function compute_data()
    df = DataFrame()
    for mean_s in [20, 50, 100, 200]
        system, initial = prepare(mean_s)
        df = vcat(df, get_results(an, 128, system, initial))
        df = vcat(df, get_results(ti, 128, system, initial))
        df = vcat(df, get_results(direct, 128, system, initial))
    end
    df
end

df.mean_s

# df = compute_data()
# using CSV
# CSV.write(projectdir("data", "estimator_variance", "estimator_comparison.csv"), df)

df["mean_s"] = df.κ ./ df.λ
df["mean_s_cat"] = string.(df.mean_s)

df2 = transform(groupby(df, [:mean_s]), :log_marginal => (x-> x .- mean(x[x.estimate == "AIS"])) => :deviation)
show(df2)

using Plots
using StatsPlots
using Statistics

@df df2 dotplot(:mean_s_cat, :deviation, group=(:estimate))

@show combine(groupby(df, [:estimate, :mean_s]), :log_marginal => mean, :log_marginal => std)

jitter_ti = randn(size(ti_samples)...)
jitter_an = randn(size(an_samples)...)
jitter_di = randn(size(di_samples)...)
p = scatter(jitter_ti, ti_samples, label="TI", xlim=(-15, 15))
scatter!(jitter_an, an_samples, label="AIS")
scatter!(jitter_di, di_samples, label="Direct MC")
plot!(p, [-3,3], mean(ti_samples) .* ones(2))
plot!(p, [-3,3], mean(an_samples) .* ones(2))
plot!(p, [-3,3], mean(di_samples) .* ones(2))

density([ti_samples, an_samples, di_samples], label=hcat("TI", "AIS", "Direct MC"))