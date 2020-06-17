using StatsFuns


function estimate_marginal_entropy(system::System, t::Matrix{<:Real}, num_responses::Integer, ::Val{Strategy}; kwargs...) where Strategy
    joint_dist = joint(system, t)
    data = []
    for i in 1:num_responses
        initial = rand(joint_dist)
        (estimate, additional_data), elapsed, bytes, gctime, memallocs = @timed estimate_marginal_density(Val(Strategy), system, t, initial; kwargs...)
        new_data = (
            estimate=estimate,
            state=initial,
            system=system,
            t_matrix=t,
            extra_data=additional_data,
            elapsed_time=elapsed,
            bytes=bytes,
            gctime=gctime,
            memallocs=memallocs,
        )
        push!(data, new_data)
    end
    data
end

signal_part(conf::GaussianMcmc.SystemConfiguration) = conf.state[begin:length(conf.state) ÷ 2]
response_part(conf::GaussianMcmc.SystemConfiguration) = conf.state[length(conf.state) ÷ 2 + 1 : end]

function dos_distributions(system::System, t::Matrix{<:Real}, initial=nothing)
    if initial === nothing
        initial = GaussianProposal(rand(joint(system, t)))
    else 
        initial = GaussianProposal(initial)
    end
    
    response = response_part(initial)
    signals = rand(prior(system, t), 100_000);

    samples = hcat(vcat.(eachcol(signals), Ref(response))...)
    sample_energies = -log_likelihood(system, t, signal=signals, response=response)

    sort(sample_energies)
end


function estimate_marginal_density(::Val{:WangLandau}, system::System, t::Matrix{<:Real}, initial::Vector{<:Real}; scale::Real, num_bins::Integer, ϵ::Real)
    dos_range = dos_distributions(system, t, initial)

    energy_bins = collect(range(dos_range[begin] * 1.0, dos_range[end], length=num_bins+1));
    energies = (energy_bins[begin+1:end] + energy_bins[begin:end-1]) ./ 2
    
    initial = GaussianProposal(initial)
    wl = WangLandau(system, t, 1.0, scale, initial, energy_bins, ϵ=ϵ)

    acceptance = []
    hist_list = []
    entr_list = []
    conf_list = []
    for (accepted, rejected) in wl
        ratio = accepted / (rejected + accepted)
        push!(acceptance, ratio)
        push!(hist_list, copy(wl.histogram))
        push!(entr_list, copy(wl.entropy))
        push!(conf_list, copy(wl.current_conf))
        println("ratio=$ratio, f=$(wl.f_param), flatness=$(flatness(wl.histogram))")
    end

    result = (
        acceptance=acceptance,
        histograms=hcat(hist_list...),
        entropies=hcat(entr_list...),
        configurations=vcat(conf_list...),
        wang_landau=wl,
    )

    log_dos = wl.entropy .- logsumexp(wl.entropy .+ log.(diff(energy_bins)))
    estimate = logsumexp(-energies .+ log_dos .+ log.(diff(energy_bins)))

    (estimate, result)
end

function estimate_marginal_density(::Val{:ThermodynamicIntegration}, system::System, t::Matrix{<:Real}, initial::Vector{<:Real}; scale::Real, num_samples::Integer, skip::Integer)
    θ = rand()
    initial = GaussianProposal(initial)
    samples, acceptance = generate_mcmc_samples(initial, num_samples, system, t; scale = scale, skip = skip, θ = θ)

    n_dim = size(t, 1)

    signal = @view samples[1:n_dim,:]
    response = @view samples[n_dim + 1:end, 1]

    ll = log_likelihood(system, t, signal=signal, response=response)

    extra_info = (
        θ=θ,
        samples=samples,
        acceptance=acceptance
    )

    (mean(ll), extra_info)
end
