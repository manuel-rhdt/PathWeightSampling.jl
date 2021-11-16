# Marginalization Strategies

The computationally most demanding part of PWS is the evaluation of the marginalization integral ``\mathcal{P}[\bm{x}]=\int\mathcal{D}[\bm{s}] \mathcal{P}[\bm{s},\bm{x}]`` which needs to be computed repeatedly for many different outputs ``\bm{x}_1,\ldots,\bm{x}_N``. Consequently the computational efficiency of the marginalization is essential for the overall simulation performance.

Marginalization is a general term to denote an operation where one or more variables are integrated out of a joint probability distribution, say ``\mathcal{P}[\bm{s},\bm{x}]``, to obtain the corresponding marginal probability distribution ``\mathcal{P}[\bm{x}]``:
```math
    \mathcal{P}[\bm{x}] = \int\mathcal{D}[\bm{s}]\ \mathcal{P}[\bm{s},\bm{x}]\,.
```

Note that in our case, ``\bm{s}`` and ``\bm{x}`` are trajectories
and that therefore the integral ``\int\mathcal{D}[\bm{s}]`` is actually a path integral.

## Equivalence of Marginalization Integrals and Free Energy Computations

In this section we establish the equivalence between computing a marginalization integral
and computing the free energy in statistical physics.

In the language of statistical physics, ``\mathcal{P}[\bm{x}]`` corresponds to the normalization constant, or partition function, of a Boltzmann distribution for the potential
```math
    \mathcal{U}[\bm{s},\bm{x}] = -\ln\mathcal{P}[\bm{s},\bm{x}] \,.
```
We interpret ``\bm{s}`` as a variable in the configuration space whereas ``\bm{x}`` is an auxiliary variable, i.e. a parameter. Note that both ``\bm{s}`` and ``\bm{x}`` still represent trajectories. For this potential, the partition function is given by
```math
    \mathcal{Z}[\bm{x}] = \int\mathcal{D}[\bm{s}]\; e^{-\mathcal{U}[\bm{s},\bm{x}]} \,.
```
The integral only runs over the configuration space, i.e. we integrate only with respect to ``\bm{s}`` but not ``\bm{x}``, which remains a parameter of the partition function.
The partition function is precisely equal to the marginal probability of the output, i.e. ``\mathcal{Z}[\bm{x}] = \mathcal{P}[\bm{x}]``, as can be verified by inserting the expression for the ``\mathcal{U}[\bm{s},\bm{x}]``.
Further, the free energy is defined as
```math
    \mathcal{F}[\bm{x}] = -\ln \mathcal{Z}[\bm{x}] = -\ln \mathcal{P}[\bm{x}]\,,
```
i.e. the computation of the free energy of the trajectory ensemble corresponding to ``\mathcal{U}[\bm{s}, \bm{x}]`` is equivalent to the computation of (the logarithm of) the marginal probability ``\mathcal{P}[\bm{x}]``.

Note that above we omitted any factors of ``k_{\mathrm{B}}T`` that are typically used in physics since temperature is irrelevant here. 
Also note that while the distribution ``\exp(-\mathcal{U}[\bm{s},\bm{x}])`` looks like the equilibrium distribution of a canonical ensemble from statistical mechanics, this does not imply that the we can only study systems in thermal equilibrium. Thus, the notation introduced in this section is nothing else but a mathematical reformulation of the marginalization integral to make the analogy to statistical physics apparent and we assign no additional meaning of the potentials and free energies introduced here.

In statistical physics it is well known that the free energy cannot be directly measured from a single simulation. Instead, one estimates the free-energy difference
```math
    \Delta\mathcal{F}[\bm{x}] = \mathcal{F}[\bm{x}] - \mathcal{F}_0[\bm{x}] = -\ln \frac{\mathcal{Z}[\bm{x}]}{\mathcal{Z}_0[\bm{x}]}
```
between the system and a reference system with known free energy ``\mathcal{F}_0[\bm{x}]``. The reference system is described by the potential ``\mathcal{U}_0[\bm{s}, \bm{x}]`` with the corresponding partition function ``\mathcal{Z}_0[\bm{x}]``. 
In our case, a natural choice of reference potential is
```math
    \mathcal{U}_0[\bm{s},\bm{x}]=-\ln\mathcal{P}[\bm{s}]
```
with  the corresponding partition function
```math
        \mathcal{Z}_0[\bm{x}]=\int\mathcal{D}[\bm{s}] \mathcal{P}[\bm{s}]=1\,.
```
This means that since ``\mathcal{P}[\bm{s}]`` is a normalized probability density function, the reference free energy is zero (``\mathcal{F}_0[\bm{x}]=-\ln\mathcal{Z}_0[\bm{x}]=0``). Hence, for the above choice of reference system, the free-energy difference is
```math
    \Delta\mathcal{F}[\bm{x}]= \mathcal{F}[\bm{x}] = -\ln\mathcal{P}[\bm{x}]\,.
```

Note that in our case the reference potential ``\mathcal{U}_0[\bm{s},\bm{x}]=-\ln\mathcal{P}[\bm{s}]`` does not depend on the output trajectory ``\bm{x}``, i.e. ``\mathcal{U}_0[\bm{s},\bm{x}]\equiv\mathcal{U}_0[\bm{s}]``. It describes a *non-interacting* version of our input-output system where the input trajectories evolve completely independently of the fixed output trajectory ``\bm{x}``. 

What is the interaction between the output ``\bm{x}`` and the input trajectory ensemble?
We define the interaction potential ``\Delta\mathcal{U}[\bm{s}, \bm{x}]`` through
```math
    \mathcal{U}[\bm{s}, \bm{x}] = \mathcal{U}_0[\bm{s}] + \Delta\mathcal{U}[\bm{s}, \bm{x}] \,.
```
The interaction potential makes it apparent that the distribution of ``\bm{s}`` trajectories corresponding to the potential ``\mathcal{U}[\bm{s}, \bm{x}]`` is biased by ``\bm{x}`` with respect to the distribution corresponding to the reference potential ``\mathcal{U}_0[\bm{s}]``.
By inserting the expressions for ``\mathcal{U}_0[\bm{s}]`` and ``\mathcal{U}[\bm{s}, \bm{x}]`` we see that
```math
    \Delta\mathcal{U}[\bm{s}, \bm{x}] = -\ln\mathcal{P}[\bm{x}|\bm{s}] \,.
```
This expression illustrates that the interaction of the output trajectory ``\bm{x}`` with the ensemble of input trajectories is characterized by the trajectory likelihood ``\mathcal{P}[\bm{x}|\bm{s}]``.
Since we can compute the trajectory likelihood from the master equation, so can we compute the interaction potential.

## Direct PWS

```@docs
DirectMCEstimate
```

The direct scheme makes it possible to compute the marginal probability ``\mathcal{P}[\bm{x}]`` for the output trajectory of an externally driven Markov jump process. Yet, due to the combinatorial explosion of possible input trajectories, the variance of the direct scheme increases exponentially with trajectory length. Hence, for complex information processing networks and long trajectories the direct estimate may incur very high computational cost. Therefore, we implemented two improved variants of PWS which allow us to study more complex information processing networks.

## RR-PWS

In Rosenbluth-Rosenbluth PWS (RR-PWS) we compute the free-energy difference ``\Delta\mathcal{F}`` between the ideal system ``\mathcal{U}_0`` and ``\mathcal{U}`` in a *single* simulation just like with the direct method.
However, instead of generating ``\bm{s}`` trajectories in an uncorrelated fashion according to ``\exp(-\mathcal{U}_0[\bm{s}])=\mathcal{P}[\bm{s}]``, we bias our sampling distribution towards ``\exp(-\mathcal{U}[\bm{s}, \bm{x}])\propto\mathcal{P}[\bm{s}|\bm{x}]`` to reduce the sampling problems found in the `DirectMCEstimate`.

```@docs
SMCEstimate
```

## TI-PWS

The third marginalization scheme is based on the analogy of marginalization integrals with reversible work computations.
As before, we view the problem of computing the marginal probability ``\mathcal{P}[\bm{x}]`` as equivalent to that of computing the free energy between ensembles that are defined by the potentials ``\mathcal{U}_0[\bm{s}, \bm{x}]`` and ``\mathcal{U}[\bm{s}, \bm{x}]``, respectively. For the TI-PWS estimate we additionally assume that there is a continuous parameter that transforms the ensemble from ``\mathcal{U}_0`` to ``\mathcal{U}``.  
Mathematically, we thus have a continuous mapping from ``\theta\in[0,1]`` to a potential ``\mathcal{U}_\theta[\bm{s},\bm{x}]`` (where ``\mathcal{U}=\mathcal{U}_1``) with a corresponding partition function 
```math
\mathcal{Z}_\theta[\bm{x}]=\int\mathcal{D}[\bm{s}]\ e^{-\mathcal{U}_\theta[\bm{s},\bm{x}]} \,.
```
For instance, for ``0\leq\theta\leq 1``, we can define our potential to be 
```math
    \mathcal{U}_\theta[\bm{s},\bm{x}]=\mathcal{U}_0[\bm{s}, \bm{x}]+\theta\,\Delta\mathcal{U}[\bm{s},\bm{x}]\,,
```
such that ``e^{-\mathcal{U}_\theta[\bm{s},\bm{x}]}=\mathcal{P}[\bm{s}]\mathcal{P}[\bm{x}|\bm{s}]^\theta``.

To derive the thermodynamic integration estimate for the free-energy difference, we first compute the derivative of ``\ln\mathcal{Z}_\theta[\bm{x}]`` with respect to ``\theta``:
```math
\begin{aligned}
    \frac{\partial}{\partial \theta} \ln\mathcal{Z}_\theta[\bm{x}] &= \frac{1}{\mathcal{Z}_\theta[\bm{x}]} \frac{\partial}{\partial \theta} \int\mathcal{D}[\bm{s}]\  e^{-\mathcal{U}_\theta[\bm{s},\bm{x}]} \\
    &= -\left\langle \frac{\partial \mathcal{U}_\theta[\bm{s},\bm{x}]}{\partial\theta} \right\rangle_\theta\\
    &= -\left\langle
    \Delta\mathcal{U}[\bm{s},\bm{x}]
    \right\rangle_\theta\,.
\end{aligned}
```
Thus, the derivative of ``\ln\mathcal{Z}_\theta[\bm{x}]`` is an average of the Boltzmann weight with respect to ``\mathcal{P}_\theta[\bm{s}|\bm{x}]`` which is the ensemble distribution of ``\bm{s}`` given by
```math
    \mathcal{P}_\theta[\bm{s}|\bm{x}] = \frac{1}{\mathcal{Z}_\theta[\bm{x}]} e^{-\mathcal{U}_\theta[\bm{s}, \bm{x}]}\,.
```
Integrating with respect to ``\theta`` leads to the formula for the free-energy difference
```math
    \Delta\mathcal{F}[\bm{x}] = -\int^1_0 \mathrm{d}\theta\ \left\langle 
    \Delta\mathcal{U}[\bm{s}, \bm{x}]
    \right\rangle_\theta
```
which is the fundamental identity underlying thermodynamic integration.

```@docs
TIEstimate
```