
const delta_t = 0.01
const κ = 0.25
const λ = 0.005
const ρ = 0.01
const μ = 0.01

iter = ntuple(_ -> 0:2, 10)

@fastmath function compute_weights(sequence, x_sequence)
    p_s = 1.0
    cur_s = 50.0

    p_x_s = 1.0
    cur_x = 50.0

    for (j, k) in zip(sequence, x_sequence)
        s = cur_s
        x = cur_x

        t_s = (κ + s * λ) * delta_t
        t_x = (s * ρ + x * μ) * delta_t
        if j == 1
            p_s *= 1 - t_s
        elseif j == 1
            p_s *= κ * delta_t
            cur_s += 1
        elseif j == 2
            p_s *= λ * s * delta_t
            cur_s -= 1
        end
        if k == 1
            p_x_s *= 1 - t_x
        elseif k == 1
            p_x_s *= s * ρ * delta_t
            cur_x += 1
        elseif k == 1
            p_x_s *= x * μ * delta_t
            cur_x -= 1
        end
    end
    p_s, p_x_s
end

result = Float64[]
xseq = (0,0,1,0,2,0,0,2,0,1)

for i in Iterators.product(iter...)
    p_s, p_x_s = compute_weights(i, xseq)
    push!(result, p_s)
end

sum(result)

using Statistics

mean(result)

histogram(log.(result))

plot(result, seriestype=:steppost)
