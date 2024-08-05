import PathWeightSampling as PWS
using Test

# create a coupled birth-death system
rates = [50.0, 1.0, 1.0, 1.0]
rstoich = [[], [1 => 1], [1 => 1], [2 => 1]]
nstoich = [[1 => 1], [1 => -1], [2 => 1], [2 => -1]]
reactions = PWS.ReactionSet(rates, rstoich, nstoich, 2)

system = PWS.MarkovJumpSystem(
    PWS.GillespieDirect(),
    reactions,
    [50, 0],
    (0.0, 20.0),
    [0, 0, 1, 2],
    BitSet([3, 4]);
    dt=0.5
)

conf = PWS.generate_configuration(system, seed=1)

alg = PWS.PERM(512)
@time perm_result = PWS.simulate(alg, conf, system; Particle=PWS.MarkovParticle)

alg_smc = PWS.SMCEstimate(512)
@time smc_result = PWS.simulate(alg_smc, conf, system; Particle=PWS.MarkovParticle)

@test all(
    map(
        (x, y) -> isapprox(x, y, rtol=0.001),
        PWS.log_marginal(perm_result),
        PWS.log_marginal(smc_result)
    )
)