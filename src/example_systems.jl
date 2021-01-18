import Catalyst: @reaction_network

function chemotaxis_system(; mean_L=20, num_receptors=10, Y_tot=50, L_timescale=1.0, LR_timescale=0.5, Y_timescale=0.1, duration=2.0)
    LR_ratio = 0.5
    Y_ratio = 0.5

    mean_LR = num_receptors * LR_ratio
    mean_R = num_receptors - mean_LR

    mean_Yp = Y_tot * Y_ratio
    mean_Y = Y_tot - mean_Yp

    sn = @reaction_network begin
        κ, ∅ --> L
        λ, L --> ∅
    end κ λ

    rn = @reaction_network begin
        ρ, L + R --> L + LR
        μ, LR --> R
    end ρ μ

    xn = @reaction_network begin
        δ, LR + Y --> Yp + LR
        χ, Yp --> Y
    end δ χ

    u0 = SA[mean_L, mean_R, mean_LR, mean_Y, mean_Yp]
    tspan = (0.0, duration)
    ps = [mean_L, 1/L_timescale]
    pr = [mean_LR / (LR_timescale * mean_R * mean_L), 1 / LR_timescale]
    px = [mean_Yp / (Y_timescale * mean_Y * mean_LR), 1 / Y_timescale]

    SRXsystem(sn, rn, xn, u0, ps, pr, px, tspan)
end