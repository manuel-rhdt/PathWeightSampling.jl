# Guide

In this section we will show step by step how PathWeightSampling.jl can be used to compute the mutual information for a simple model of gene expression.

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
The first two reactions specify the evolution of the input signal `S`, and last two reactions specify the evolution of the output `X`. Thus, both the input and output signal are modeled as a simple birth-death process, however the birth rate of `X` increases with higher copy numbers of  `S`.

The first step is to create the system that we are going to use. The simple gene expression model shown above is already included as an example in PathWeightSampling.jl and can be directly used as follows:
```@example 1
using PathWeightSampling

system = PathWeightSampling.gene_expression_system()
```
The result is a `system` consisting of the 4 reactions mentioned above and default values for the initial condition and the parameters that specify the reaction rates.

## Generating and Plotting Trajectories

We can generate a _configuration_ of this system. A configuration is a combination of an input trajectory and an output trajectories. Using `generate_configuration` we can create a configuration by first simulating an input trajectory and then use that input trajectory to simulate a corresponding output trajectory.
```@example 1
conf = generate_configuration(system)
```

Let us plot the generated configuration:
```@example 1
ENV["GKSwstype"] = "100" # hide
using Plots
plot(conf)
savefig("plot1.svg"); nothing # hide
```
![](plot1.svg)

We see a plot of the generated input and output trajectories that make up the configuration.

The individual trajectories of the configuration can also be accessed directly:
```@example 1
input_traj = conf.s_traj
output_traj = conf.x_traj
p1 = plot(input_traj, label="input")
p2 = plot(output_traj, label="output")
plot(p1, p2, layout = (2, 1))
savefig("plot2.svg"); nothing # hide
```
![](plot2.svg)

## Computing the Trajectory Mutual Information

For our system we can compute the trajectory mutual information straightforwardly. 
```@example 1
result = PathWeightSampling.mutual_information(system, DirectMCEstimate(256), num_samples=100)
nothing # hide
```

This performs a full PWS Monte Carlo simulation and displays a progress bar during the computation. Naturally, the `PWS.mutual_information`
takes the `system` as its first argument. The second argument is an object specifying the *marginalization* algorithm to use for
computing the marginal trajectory probability. Here we chose the simple brute-force `DirectMC` algorithm with ``M=256`` samples.
Thus, we compute a "Direct PWS" estimate. The final keyword argument is the overall number of Monte Carlo samples to use for estimating the
mutual information. This is the number of samples taken in the *outer* Monte Carlo simulation as opposed to the ``M=256`` samples taken in the *inner* Monte Carlo loop.

`result` is a `DataFrame` containing the simulation results. We can display the individual Monte Carlo samples:
```@example 1
plot(
    system.dtimes, 
    result.MutualInformation, 
    color=:black, 
    linewidth=0.2, 
    legend=false, 
    xlabel="trajectory duration", 
    ylabel="mutual information (nats)"
)
savefig("plot3.svg"); nothing # hide
```

![](plot3.svg)

The final Monte Carlo estimate is simply the `mean` of the individual samples:
```@example 1
using Statistics
plot(
    system.dtimes, 
    mean(result.MutualInformation), 
    color=:black, 
    linewidth=2, 
    legend=false,
    xlabel="trajectory duration",
    ylabel="mutual information (nats)"
)
savefig("plot4.svg"); nothing # hide
```

![](plot4.svg)

Note that since we only used 100 MC samples the fluctuation of the result is relatively large. To judge the statistical error due to the number of Monte Carlo samples, we can additionally plot error bars. A common error measure in Monte Carlo simulations is the "standard error of the mean", defined as the standard deviation divided by the square root of the number of samples. We use this method to draw error bars.

```@example 1
sem(x) = std(x) / sqrt(length(x))
plot(
    system.dtimes, 
    mean(result.MutualInformation),
    yerr=sem(result.MutualInformation), 
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

So far we computed the mutual information using the brute-force *Direct PWS* algorithm. However, we can choose a different approach to perform
the marginalization integrals. To change the marginalization strategy we simply pass a different `algorithm` as the second argument of `PWS.mutual_information`. The possible choices for the marginalization strategy are

- `DirectMCEstimate(m)`: The simple brute force marginalization using a Direct Monte Carlo estimate. The integer `m` specifies the number of samples to use per brute-force computation. This method works well for short trajectories but becomes exponentially worse for longer trajectories.
- `SMCEstimate(m)`: Improved computation of marginalization integrals using a sequential Monte Carlo technique (specifically using a particle filter). The integer `m` specifies the number of "particles" that are being propagated simultaneously. This method works much better than the `DirectMCEstimate` for long trajectories.
- `TIEstimate(burn_in, integration_nodes, num_samples)`: Use thermodynamic integration to compute the marginalization integrals. This will set up a number of MCMC simulations in path-space to perform the TI integral. `burn_in` specifies the number of initial samples from the MCMC simulation to be discarded, `integration_nodes` specifies the number of points to use in the Gaussian quadrature, and `num_samples` specifies the number of MCMC samples per integration node to generate.
- `AnnealingEstimate(subsample, num_temps, num_samples)`: Use annealed importance sampling to compute the marginalization integrals. This technique is very similar to thermodynamic integration and also uses MCMC simulations in path space. `subsample` specifies the number of Metropolis trials to perform before recording a new MCMC sample. `num_temps` sets how many different "temperatures" should be used for the annealing. `num_samples` is the number of MCMC samples to use per temperature setting.

We can compute the mutual information using each of these strategies and compare the results:

```@example 1
strategies = [
    DirectMCEstimate(128), 
    SMCEstimate(128), 
    TIEstimate(0, 8, 16), 
    # AnnealingEstimate(0, 128, 1)
]
results = [PathWeightSampling.mutual_information(system, strat, num_samples=100, progress=false) for strat in strategies]

plot()
for (strat, r) in zip(strategies, results)
    plot!(
        system.dtimes, 
        mean(r.MutualInformation),
        label=PathWeightSampling.name(strat),
        xlabel="trajectory duration",
        ylabel="mutual information (nats)"
    )
end

savefig("plot6.svg"); nothing # hide
```

![](plot6.svg)

## API Summary

Thus, the core function to estimate the trajectory mutual information is `PathWeightSampling.mutual_information`. A complete description of its arguments and return value is given below.
```@docs
PathWeightSampling.mutual_information
```
