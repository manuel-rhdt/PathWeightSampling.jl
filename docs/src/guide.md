# Guide

This guide computes the mutual information for a simple gene-expression model.

## Setting Up the System

The model considered in this example consists of four reactions:
```
                κ
reaction 1:  ∅ ---> S

                λ
reaction 2:  S ---> ∅

                ρ
reaction 3:  S ---> S + X

                μ
reaction 4:  X ---> ∅
```
The first two reactions govern the input signal `S`; the last two govern the output `X`. Both follow birth-death dynamics, but the birth rate of `X` increases with the copy number of `S`.

This gene-expression model is included in PathWeightSampling.jl. Create it with:
```@example 1
import PathWeightSampling as PWS

system = PWS.gene_expression_system()
```
This returns a `system` containing the four reactions, default initial conditions, and default reaction-rate parameters.

## Generating and Plotting Trajectories

A _configuration_ pairs an input trajectory with an output trajectory. `generate_configuration` simulates the input first, then uses it to simulate the corresponding output.
```@example 1
conf = PWS.generate_configuration(system)
```

Plot the configuration with:
```@example 1
ENV["GKSwstype"] = "100" # hide
using Plots
plot(conf)
savefig("plot1.svg"); nothing # hide
```
![](plot1.svg)

The plot shows the input and output trajectories that make up the configuration.

The individual trajectories of the configuration can also be accessed directly:
```@example 1
input_traj = conf.traj[1, :]
output_traj = conf.traj[2, :]
p1 = plot(conf.discrete_times, input_traj, label="input")
p2 = plot(conf.discrete_times, output_traj, label="output")
plot(p1, p2, layout = (2, 1))
savefig("plot2.svg"); nothing # hide
```
![](plot2.svg)

## Computing the Trajectory Mutual Information

Compute the trajectory mutual information with:
```@example 1
data = PWS.mutual_information(system, PWS.DirectMCEstimate(256), num_samples=100)
nothing # hide
```

This runs a PWS Monte Carlo simulation and displays a progress bar. `PWS.mutual_information` takes the `system` as its first argument. The second argument selects the *marginalization* algorithm used to compute the marginal trajectory probability. Here, `DirectMCEstimate(256)` gives a "Direct PWS" estimate using ``M=256`` brute-force samples.

The `num_samples` keyword controls the number of samples in the *outer* Monte Carlo simulation, as distinct from the ``M=256`` samples in the *inner* Monte Carlo loop.

`data.result` is a `DataFrame` containing the simulation results. Plot the individual Monte Carlo samples with:
```@example 1
df = data.result
scatter(
    df.time, 
    df.MutualInformation, 
    color=:black, 
    legend=false, 
    xlabel="trajectory duration", 
    ylabel="mutual information (nats)"
)
savefig("plot3.svg"); nothing # hide
```

![](plot3.svg)

The Monte Carlo estimate is the mean of these samples:
```@example 1
using Statistics
using DataFrames
df = combine(groupby(data.result, :time), :MutualInformation => mean => :MI)
plot(
    df.time, 
    df.MI, 
    color=:black, 
    linewidth=2, 
    legend=false,
    xlabel="trajectory duration",
    ylabel="mutual information (nats)"
)
savefig("plot4.svg"); nothing # hide
```

![](plot4.svg)

With only 100 Monte Carlo samples, the estimates fluctuate noticeably. Error bars show the standard error of the mean: the standard deviation divided by the square root of the number of samples.

```@example 1
sem(x) = std(x) / sqrt(length(x))
df = combine(groupby(data.result, :time), :MutualInformation => mean => :MI, :MutualInformation => sem => :Err)
plot(
    df.time, 
    df.MI,
    yerr=df.Err, 
    color=:black, 
    linewidth=2, 
    legend=false,
    xlabel="trajectory duration",
    ylabel="mutual information (nats)"
)
savefig("plot5.svg"); nothing # hide
```

![](plot5.svg)

## More Advanced Marginalization Strategies

The examples above used the brute-force *Direct PWS* algorithm. To choose another approach to the marginalization integrals, pass a different estimator as the second argument of `PWS.mutual_information`:

- `DirectMCEstimate(m)`: The simple brute force marginalization using a Direct Monte Carlo estimate. The integer `m` specifies the number of samples to use per brute-force computation. This method works well for short trajectories but becomes exponentially worse for longer trajectories.
- `SMCEstimate(m)`: Improved computation of marginalization integrals using a sequential Monte Carlo technique (specifically using a particle filter). The integer `m` specifies the number of "particles" that are being propagated simultaneously. This method works much better than the `DirectMCEstimate` for long trajectories.
- `TIEstimate(burn_in, integration_nodes, num_samples)`: Use thermodynamic integration to compute the marginalization integrals. This will set up a number of MCMC simulations in path-space to perform the TI integral. `burn_in` specifies the number of initial samples from the MCMC simulation to be discarded, `integration_nodes` specifies the number of points to use in the Gaussian quadrature, and `num_samples` specifies the number of MCMC samples per integration node to generate.
- `AnnealingEstimate(subsample, num_temps, num_samples)`: Use annealed importance sampling to compute the marginalization integrals. This technique is very similar to thermodynamic integration and also uses MCMC simulations in path space. `subsample` specifies the number of Metropolis trials to perform before recording a new MCMC sample. `num_temps` sets how many different "temperatures" should be used for the annealing. `num_samples` is the number of MCMC samples to use per temperature setting.

We can compute the mutual information using each of these strategies and compare the results:

```@example 1
strategies = [
    PWS.DirectMCEstimate(128),
    PWS.SMCEstimate(128),
    PWS.PERM(16)
]
results = [PWS.mutual_information(system, strat, num_samples=100, progress=false) for strat in strategies]

plot()
for (strat, r) in zip(strategies, results)
    res = combine(groupby(r.result, :time), :MutualInformation => mean => :MI)
    plot!(
        res.time, 
        res.MI,
        label=PWS.name(strat),
        xlabel="trajectory duration",
        ylabel="mutual information (nats)"
    )
end

savefig("plot6.svg"); nothing # hide
```

![](plot6.svg)

## API Summary

The core function for estimating trajectory mutual information is `PathWeightSampling.mutual_information`. Its arguments and return value are documented below.
```@docs
PathWeightSampling.mutual_information
```
