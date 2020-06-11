using Printf

flatness(histogram::Vector) = minimum(histogram) / maximum(histogram)

mutable struct WangLandau{Conf <: SystemConfiguration}
    system::System
    t::Matrix{Float64}
    f_param::Float64
    entropy::Vector{Float64}
    histogram::Vector{UInt64}
    scale::Float64
    initial::Conf
    energy_bins::Vector{Float64}
    ϵ::Float64
end

function perform(iter::WangLandau)
    joint = FastMvNormal(corr_z(iter.system, iter.t))

    current_energy = energy(iter.initial, joint)
    current_energy_bin = searchsortedfirst(iter.energy_bins, current_energy) - 1
    
    accepted = 0
    rejected = 0
    
    current_conf = copy(iter.initial)
    new_conf = similar(iter.initial)
    
    while iter.f_param > iter.ϵ
        propose_conf!(new_conf, current_conf, iter.scale)
        new_energy = energy(new_conf, joint)
        
        new_energy_bin = searchsortedfirst(iter.energy_bins, new_energy) - 1

        if new_energy_bin >= 1 && 
            new_energy_bin <= length(iter.entropy) && 
            rand() < exp(iter.entropy[current_energy_bin] - iter.entropy[new_energy_bin])
            accepted += 1
            current_energy = new_energy
            current_energy_bin = new_energy_bin
            copy!(current_conf, new_conf)
        else
            rejected += 1
        end
        
        iter.histogram[current_energy_bin] += 1
        iter.entropy[current_energy_bin] += iter.f_param

        if flatness(iter.histogram) >= 0.95
            @printf("update f: %f -> %f\n", iter.f_param, iter.f_param * 0.5)
            fill!(iter.histogram, 0)
            iter.f_param *= 0.5
        end
    end
end