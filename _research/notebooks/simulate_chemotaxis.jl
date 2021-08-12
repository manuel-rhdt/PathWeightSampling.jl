### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 0026826c-997f-4ed7-8c0b-c54d199010a9
using Revise

# ╔═╡ 3dd034e0-95f0-11eb-0285-7fa1f7afbcae
begin
	import GaussianMcmc
	using GaussianMcmc: TrajectoryCallback, Trajectory, SMCEstimate, DirectMCEstimate, marginal_configuration, MarginalEnsemble, ConditionalEnsemble, gene_expression_system, generate_configuration, mutual_information, logpdf, simulate, cooperative_chemotaxis_system, reaction_network, collect_trajectory, log_marginal
	using StaticArrays
	import Catalyst
	using DiffEqBase
	using DiffEqJump
	import ModelingToolkit
	using Plots
	using Transducers
	using Statistics
end

# ╔═╡ 11b80f90-068e-4ebd-8015-d1a92218928e
n_clusters=300

# ╔═╡ bbafdb7c-a080-4ab9-ad43-fa124be6da1c
system = cooperative_chemotaxis_system(; n_clusters, phosphorylate=0.001, dephosphorylate=0.5, dtimes=0.0:0.5:100)

# ╔═╡ 5548384b-cd4e-437b-836f-8b44715b9328
conf = generate_configuration(system)

# ╔═╡ b46de91d-350d-48f6-bdf3-d0bc9fcf4632
begin
	plot(conf.s_traj |> Map((u,t,i)::Tuple -> (u ./ 100, t, i)) |> collect_trajectory, label="Ligand concentration")
	plot!(GaussianMcmc.active_receptors(conf, system) |> Map((u,t,i)::Tuple -> (u ./ n_clusters, t, i)) |> collect_trajectory, label="active fraction")
	plot!(conf.x_traj |> Map((u,t,i)::Tuple -> (u[SA[1]] ./ 10000, t, i)) |> collect_trajectory)
	plot!(fmt=:png, legend=:bottomright, dpi=300)
end

# ╔═╡ 5529e55a-9436-44b1-8268-46e054dca1d1
alg = SMCEstimate(16)

# ╔═╡ d3f75f23-4c6a-4a7d-9318-dcea0382bcf3
mi = mutual_information(system, alg, num_responses=10)

# ╔═╡ 6a41ab5b-0c2c-4acd-91d4-73f8b163221a
plot(system.dtimes, mi.MutualInformation, color=:gray, legend=false)

# ╔═╡ 6b2359a9-24b5-4a58-a10b-7f8729c3e4e9
plot(system.dtimes, mean(mi.MutualInformation), ribbon=sqrt.(var(mi.MutualInformation)./size(mi, 1)), label="SMC", ylabel="Path mutual information", xlabel="Trajectory length")

# ╔═╡ Cell order:
# ╠═0026826c-997f-4ed7-8c0b-c54d199010a9
# ╠═3dd034e0-95f0-11eb-0285-7fa1f7afbcae
# ╠═11b80f90-068e-4ebd-8015-d1a92218928e
# ╠═bbafdb7c-a080-4ab9-ad43-fa124be6da1c
# ╠═5548384b-cd4e-437b-836f-8b44715b9328
# ╠═b46de91d-350d-48f6-bdf3-d0bc9fcf4632
# ╠═d3f75f23-4c6a-4a7d-9318-dcea0382bcf3
# ╠═6a41ab5b-0c2c-4acd-91d4-73f8b163221a
# ╠═6b2359a9-24b5-4a58-a10b-7f8729c3e4e9
# ╠═5529e55a-9436-44b1-8268-46e054dca1d1
