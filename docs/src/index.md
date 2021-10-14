
# Background

Information is a fundamental resource in complex systems at all scales, from bacterial signaling networks to artificial neural networks. 
These systems receive input signals which are analyzed, filtered, transcoded, or otherwise transformed to yield an output signal.
The generic mathematical framework to study the flow of information in these systems is *information theory* which defines a central quantity to quantify the amount of information that the output contains about the input: the *mutual information*. Specifically, this measure quantifies the number of distinct mappings between input and output that can be distinguished uniquely and is thus a measure of the fidelity of the input-output relationship.

Mathematically, the mutual information between two random variables, $S$ and $X$ is defined as
$$
\mathrm{I}(S, X) = \sum_{s\in S, x\in X} \mathrm{P}(s, x) \ln\frac{\mathrm{P}(s, x)}{\mathrm{P}(s) \mathrm{P}(x)} \,.
$$

However, the mutual information between scalar random variables $S$ and $X$ does not quantify the *rate* of information transmission. Indeed, most systems receive time-dependent input signals, i.e. a sequence of messages over time, which are transformed into a time-dependent output signal. 
In most systems a given input message is not independent from previous input messages. These autocorrelations within the input reduce the rate at which information is received. Additionally, correlations within the output signal can also reduce the information transmission rate. Hence, to quantify the rate of information transmission, the instantaneous mutual information between scalar random variables $S$ and $X$ is not sufficient, and we require a better mutual information measure.

The solution to this problem is to compute the mutual information between entire trajectories of input and output, not between input and output values at given time points. In this way, correlations within the input and the output are taken into account when computing the mutual information. The mathematical form of the trajectory mutual information is analogous to the scalar case:
$$
\mathrm{I}(\bm{S}, \bm{X}) = \sum_{\bm{s}\in \bm{S}, \bm{x}\in \bm{X}} \mathrm{P}(\bm{s}, \bm{x}) \ln\frac{\mathrm{P}(\bm{s}, \bm{x})}{\mathrm{P}(\bm{s}) \mathrm{P}(\bm{x})}
$$
but now the sum runs over all possible input and output trajectories (which are denoted using bold symbols), not unlike a path integral. 
The information transmission rate $\mathrm{R}(\bm{S}, \bm{X})$ is then defined as the asymptotic rate at which the trajectory mutual information increases with the duration of the input and output trajectories, i.e.
$$
\mathrm{R}(\bm{S}, \bm{X}) = \lim_{T\rightarrow\infty} \frac{\mathrm{d}\mathrm{I}(\bm{S}_T, \bm{X}_T)}{\mathrm{d}T}
$$
where $\bm{S}_T$ and $\bm{X}_T$ are trajectory-valued random variables of trajectories with duration $T$.

PWS is a method to compute the mutual information between input and output trajectories for systems described by a master equation. 