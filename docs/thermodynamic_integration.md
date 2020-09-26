# Thermodynamic Integral

- From the master equation we can compute $\mathrm P(\mathbf s)$ and $\mathrm P(\mathbf x|\mathbf s)$
- the mutual information rate is defined as the asymptotic increase of mutual information between trajectories of duration $T$.
- $\mathrm H(\mathcal X) = \int\mathrm d\mathbf x\ \mathrm P(\mathbf x)\ln\mathrm P(\mathbf x)$
- $\mathrm P(\mathbf x)$ can be regarded as the normalization constant for $\mathrm P(\mathbf s|\mathbf x)$

## Creating a Markov Chain of Entire Trajectories

$\mathrm P(\mathbf s,\mathbf x)$ is a probability density defined for entire trajectories $\mathbf s = \{(s_1, t_1),\ldots,(s_N,t_N)\}$ and $\mathbf x = \{(x_1, t_1),\ldots,(x_M,t_M)\}$. Using the Metropolis acceptance criterion we can create a Markov chain for trajectories $\mathbf s$ with an arbitrary stationary distribution $p_S(\mathbf s)$, provided that we can compute a function $f_S(\mathbf s)$ that is proportional to $p_S$.

The goal of the Metropolis-Hastings algorithm is to build a chain of trajectories $\mathbf s_1,\ldots,\mathbf s_K$ that are distributed according to $p_S(\mathbf s)$. A step in the Metropolis-Hastings algorithm involves generating a correlated trajectory $\mathbf s^\prime$ from an arbitrarily chosen trial distribution $\mathrm P(\mathbf s_n\rightarrow\mathbf s^\prime)$. We accept the new trajectory $\mathbf s^\prime$ with probability $$\mathrm A(\mathbf s^\prime, \mathbf s_n) = \min\left\{ 1,  \frac{f_S(\mathbf s^\prime)}{f_S(\mathbf s)} \frac{\mathrm P(\mathbf s^\prime \rightarrow \mathbf s_n)}{\mathrm P(\mathbf s_n\rightarrow\mathbf s^\prime)} \right\}.$$ 
In the case of acceptance we set $\mathbf s_{n+1} = \mathbf s^\prime$ and otherwise we just have $\mathbf s_{n+1}=\mathbf s_n.$ This acceptance criterion ensures that the chain converges towards the desired stationary distribution.

While the choice of the trial distribution $\mathrm P(\mathbf s_n\rightarrow\mathbf s^\prime)$ is in principle arbitrary
