### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 0026826c-997f-4ed7-8c0b-c54d199010a9
using Revise

# ╔═╡ 3dd034e0-95f0-11eb-0285-7fa1f7afbcae
begin
	import GaussianMcmc
	using GaussianMcmc: TrajectoryCallback, Trajectory, SMCEstimate, DirectMCEstimate, marginal_configuration, MarginalEnsemble, ConditionalEnsemble, gene_expression_system, generate_configuration, logpdf, simulate, cooperative_chemotaxis_system, reaction_network, collect_trajectory, log_marginal
	using StaticArrays
	import Catalyst
	using DiffEqBase
	using DiffEqJump
	import ModelingToolkit
	using Plots
	using Transducers
end

# ╔═╡ 11b80f90-068e-4ebd-8015-d1a92218928e
n_clusters=300

# ╔═╡ bbafdb7c-a080-4ab9-ad43-fa124be6da1c
@time system = cooperative_chemotaxis_system(; n_clusters, phosphorylate=0.001, dephosphorylate=0.5, dtimes=0.0:0.05:100)

# ╔═╡ 5548384b-cd4e-437b-836f-8b44715b9328
conf = generate_configuration(system)

# ╔═╡ b46de91d-350d-48f6-bdf3-d0bc9fcf4632
begin
	plot(conf.s_traj |> Map((u,t,i)::Tuple -> (u ./ 100, t, i)) |> collect_trajectory, label="Ligand concentration")
	plot!(GaussianMcmc.active_receptors(conf, system) |> Map((u,t,i)::Tuple -> (u ./ n_clusters, t, i)) |> collect_trajectory, label="active fraction")
	plot!(conf.x_traj |> Map((u,t,i)::Tuple -> (u[SA[1]] ./ 10000, t, i)) |> collect_trajectory)
	plot!(fmt=:png, legend=:bottomright)
end

# ╔═╡ 2ccbedb7-4f8e-46f0-90a0-fe8ff7007f7b
begin
	cens = ConditionalEnsemble(system)
	mens = MarginalEnsemble(system)
	mconf = marginal_configuration(conf)
end

# ╔═╡ 5529e55a-9436-44b1-8268-46e054dca1d1
alg = SMCEstimate(64)

# ╔═╡ 6571b919-f584-4f89-aa35-b26a553b2f07
result = simulate(alg, conf, cens)

# ╔═╡ 15ec2f2c-73f6-4318-8e3b-461d49f8fdbc
mresult = simulate(alg, mconf, mens)

# ╔═╡ f9244061-34d5-4a75-a60a-9c9dbf17c1f5
plot(system.dtimes, log_marginal(result) - log_marginal(mresult))

# ╔═╡ 1f99a198-e6bc-4a4c-8d32-7b0e843e00ac
log_marginal(result)

# ╔═╡ Cell order:
# ╠═0026826c-997f-4ed7-8c0b-c54d199010a9
# ╠═3dd034e0-95f0-11eb-0285-7fa1f7afbcae
# ╠═11b80f90-068e-4ebd-8015-d1a92218928e
# ╠═bbafdb7c-a080-4ab9-ad43-fa124be6da1c
# ╠═5548384b-cd4e-437b-836f-8b44715b9328
# ╠═b46de91d-350d-48f6-bdf3-d0bc9fcf4632
# ╠═2ccbedb7-4f8e-46f0-90a0-fe8ff7007f7b
# ╠═5529e55a-9436-44b1-8268-46e054dca1d1
# ╠═6571b919-f584-4f89-aa35-b26a553b2f07
# ╠═15ec2f2c-73f6-4318-8e3b-461d49f8fdbc
# ╠═f9244061-34d5-4a75-a60a-9c9dbf17c1f5
# ╠═1f99a198-e6bc-4a4c-8d32-7b0e843e00ac
