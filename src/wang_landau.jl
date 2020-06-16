flatness(histogram::Vector{<:Real}) = minimum(histogram) / mean(histogram)
is_flat(histogram::Vector{<:Real}, threshhold::Real) = flatness(histogram) >= threshhold

mutable struct WangLandau{Conf <: SystemConfiguration}
    system::System
    t::Matrix{Float64}
    f_param::Float64
    entropy::Vector{Float64}
    histogram::Vector{UInt64}
    scale::Float64
    current_conf::Conf
    energy_bins::Vector{Float64}
    skip::Integer
    ϵ::Float64
end

function WangLandau(system::System, 
        t::AbstractMatrix{<:Real}, 
        f_param::Real, 
        scale::Real,
        initial::SystemConfiguration, 
        energy_bins::Vector{<:Real}, 
        ϵ::Real = 1e-4, 
        skip::Integer = 100_000
    )
    num_bins = length(energy_bins) - 1
    WangLandau(system, t, f_param, zeros(num_bins), zeros(UInt64, num_bins), scale, initial, energy_bins, skip, ϵ)
end


Base.eltype(::Type{WangLandau{T}}) where T = Tuple{UInt,UInt}
Base.IteratorSize(::Type{WangLandau{T}}) where T = Base.SizeUnknown()

function Base.iterate(iter::WangLandau{T} where T)
    n_dim = size(iter.t, 1)

    prior = FastMvNormal(corr_ss(iter.system).(iter.t))
    joint = FastMvNormal(corr_z(iter.system, iter.t))

    current_dens = logpdf(prior, @view iter.current_conf.state[1:n_dim])
    current_energy = -(logpdf(joint, iter.current_conf.state) - current_dens)
    current_energy_bin = searchsortedfirst(iter.energy_bins, current_energy) - 1

    Base.iterate(iter, (current_dens, current_energy, current_energy_bin, prior, joint))
end


function Base.iterate(iter::WangLandau{T} where T, state)
    (current_dens, current_energy, current_energy_bin, prior, joint) = state

    n_dim = size(iter.t, 1)

    accepted::UInt = 0
    rejected::UInt = 0
    
    new_conf = similar(iter.current_conf)
    
    while iter.f_param >= iter.ϵ
        propose_conf!(new_conf, iter.current_conf, iter.scale)

        new_dens = logpdf(prior, @view new_conf.state[1:n_dim])
        new_energy = -(logpdf(joint, new_conf.state) - new_dens)
        
        new_energy_bin = searchsortedfirst(iter.energy_bins, new_energy) - 1

        if new_energy_bin >= 1 && 
            new_energy_bin <= length(iter.entropy) && 
            rand() < exp(iter.entropy[current_energy_bin] - iter.entropy[new_energy_bin] + new_dens - current_dens)
            accepted += 1
            current_dens = new_dens
            current_energy = new_energy
            current_energy_bin = new_energy_bin
            iter.current_conf.state .= new_conf.state
        else
            rejected += 1
        end
        
        iter.histogram[current_energy_bin] += 1
        iter.entropy[current_energy_bin] += iter.f_param

        if is_flat(iter.histogram, 0.95)
            fill!(iter.histogram, 0)
            iter.f_param *= 0.5
        end

        if (accepted + rejected) % iter.skip == 0
            return ((accepted, rejected), (current_dens, current_energy, current_energy_bin, prior, joint))
        end
    end

    return nothing
end