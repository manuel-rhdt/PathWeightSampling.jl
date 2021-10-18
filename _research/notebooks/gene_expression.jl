### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 98d66a30-c818-4e3d-ab62-69ebe6628815
using CairoMakie

# ╔═╡ 56ac4db9-96b4-4dfa-bb2b-0c1aa3cd551d
using Statistics

# ╔═╡ dfc116e7-606b-41cc-b6e9-c3f9ea336496
import PWS

# ╔═╡ 2737ac8b-c736-4a8f-a965-215b56788c05
system = PWS.gene_expression_system(dtimes=0:0.1:5)

# ╔═╡ 06ecfe25-740c-4267-9a2e-8ab512ba0562
md"""
# Direct Monte Carlo Estimate
"""

# ╔═╡ 86b9980f-3da8-4c49-8791-4045fff3b283
num_responses = 10^3

# ╔═╡ e6659282-7226-4b8b-a747-6478c1e2eac0
directmc = PWS.DirectMCEstimate(100)

# ╔═╡ 02932bf1-28dd-45e1-89d3-6e5b63d8f34a
directmc_result = PWS.mutual_information(system, directmc; num_responses)

# ╔═╡ 77d74d1d-59f6-4dc4-a79e-8a49954de27d
md"""
# Particle Filter Estimate
"""

# ╔═╡ 9bbc49c1-6b3a-4a3a-9f7b-b149bb00b44f
smc = PWS.SMCEstimate(100)

# ╔═╡ 2de7c4c7-906c-4035-a0b5-d459e11229a4
smc_result = PWS.mutual_information(system, smc; num_responses)

# ╔═╡ 3a8f3d25-01b2-40fb-a367-0245a0e120d4
md"# Thermodynamic Integration Estimate"

# ╔═╡ 52f1d352-7087-459d-b5c2-ec2fffb3f4fb
ti = PWS.TIEstimate(0, 20, 20)

# ╔═╡ f4262c1a-4df1-475c-8c3e-0a6a6f2c01b6
ti_result = PWS.mutual_information(system, ti; num_responses)

# ╔═╡ 5246f4b7-54c5-447b-b600-226357019c1e
md"# Comparison"

# ╔═╡ 5759fff6-d196-4e25-af7f-1d05b88ea87f
function plot_comparison(results...)
	fig = Figure()
	ax1 = Axis(fig[1, 1])
	ax2 = Axis(fig[2, 1])
	for r in results
		mi = mean(r.MutualInformation)
		error = 3*(std(r.MutualInformation) / sqrt(num_responses))
		# band!(system.dtimes, mi - error/2, mi + error/2)
		lines!(ax1, system.dtimes, mi)
		lines!(ax2, system.dtimes, mi ./ system.dtimes)
	end
	fig
end

# ╔═╡ a403bac4-2dec-431a-806f-5444a577d931
plot_comparison(directmc_result, smc_result)

# ╔═╡ fb71e194-8f76-46a9-8717-02de81272b3f
md"# Helper Functions"

# ╔═╡ 6ad0591a-15b5-436b-a812-1834ae68f503
function plot_result(result)
	fig = Figure()
	Axis(fig[1, 1])
	for mi in result.MutualInformation
		lines!(system.dtimes, mi, color=:grey, linewidth=0.3)
	end
	lines!(system.dtimes, mean(result.MutualInformation), linewidth=5)
	fig
end

# ╔═╡ 09335008-07b8-4905-a028-218d5d5020bb
plot_result(directmc_result)

# ╔═╡ d41eb941-f1b2-40bc-a0c6-0a3f82651796
plot_result(smc_result)

# ╔═╡ ad6f634d-a4f6-4b34-97e1-bad2c688ef98
plot_result(ti_result)

# ╔═╡ Cell order:
# ╠═dfc116e7-606b-41cc-b6e9-c3f9ea336496
# ╠═2737ac8b-c736-4a8f-a965-215b56788c05
# ╟─06ecfe25-740c-4267-9a2e-8ab512ba0562
# ╠═86b9980f-3da8-4c49-8791-4045fff3b283
# ╠═e6659282-7226-4b8b-a747-6478c1e2eac0
# ╠═02932bf1-28dd-45e1-89d3-6e5b63d8f34a
# ╠═09335008-07b8-4905-a028-218d5d5020bb
# ╟─77d74d1d-59f6-4dc4-a79e-8a49954de27d
# ╠═9bbc49c1-6b3a-4a3a-9f7b-b149bb00b44f
# ╠═2de7c4c7-906c-4035-a0b5-d459e11229a4
# ╠═d41eb941-f1b2-40bc-a0c6-0a3f82651796
# ╟─3a8f3d25-01b2-40fb-a367-0245a0e120d4
# ╠═52f1d352-7087-459d-b5c2-ec2fffb3f4fb
# ╠═f4262c1a-4df1-475c-8c3e-0a6a6f2c01b6
# ╠═ad6f634d-a4f6-4b34-97e1-bad2c688ef98
# ╟─5246f4b7-54c5-447b-b600-226357019c1e
# ╠═5759fff6-d196-4e25-af7f-1d05b88ea87f
# ╠═a403bac4-2dec-431a-806f-5444a577d931
# ╟─fb71e194-8f76-46a9-8717-02de81272b3f
# ╠═98d66a30-c818-4e3d-ab62-69ebe6628815
# ╠═56ac4db9-96b4-4dfa-bb2b-0c1aa3cd551d
# ╠═6ad0591a-15b5-436b-a812-1834ae68f503
