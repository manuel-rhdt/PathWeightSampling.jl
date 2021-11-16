
# Systems

## Example Systems included in PWS.jl

```@docs
PWS.gene_expression_system
PWS.cooperative_chemotaxis_system
```

## Create a New System

To create a new system, you can make either an object of type `SimpleSystem` or
`ComplexSystem` from scratch. The difference between the two is that a 
`ComplexSystem` represents a model that includes *latent variables* which need to be integrated
out to compute the mutual information. A `SimpleSystem` represents a model without
any latent variables.

Both types of systems are constructed using [Catalyst](https://catalyst.sciml.ai/)'s representation
of a reaction network (i.e. from objects of type `ReactionSystem`). Those reaction systems can be
constructed using the API from Catalyst.jl.


```@docs
PWS.SimpleSystem
PWS.ComplexSystem
```
