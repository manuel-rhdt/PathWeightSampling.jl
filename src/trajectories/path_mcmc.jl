
function cartesian_product(args...)
    n = length(args)
    indices = fill(1, n)
    lengths = [length(arg) for arg in args]

    if n == 0 || any(x -> x==0, lengths)
        return nothing
    end

    result = [tuple((args[i][indices[i]] for i in 1:n)...)]
    indices[end] += 1

    while indices[1] <= lengths[1]
        push!(result, tuple((args[i][indices[i]] for i in 1:n)...))

        # Increment indices
        j = n
        while j >= 1
            indices[j] += 1

            if indices[j] <= lengths[j] || j <= 1
                break
            end

            indices[j] = 1
            j -= 1
        end
    end

    return result
end

function homogeneous_poisson_process(rate, duration)
    t = 0.0  # Current time
    events = Float64[]  # List to store event timestamps

    while t < duration
        # Sample inter-arrival time from exponential distribution
        dt = randexp() / rate

        if t + dt <= duration
            t += dt
            push!(events, t)
        else
            break
        end
    end

    return events
end

function forward_backward(observations, transition_matrix, emission_matrix, initial_distribution)
    n_states = size(transition_matrix, 1)
    n_obs = length(observations)

    # Forward pass
    forward = zeros(n_states, n_obs)
    forward[:, 1] = initial_distribution .* emission_matrix[:, observations[1]]

    for t in 2:n_obs
        for j in 1:n_states
            forward[j, t] = emission_matrix[j, observations[t]] * sum(transition_matrix[:, j] .* forward[:, t-1])
        end
    end

    # Backward pass
    backward = zeros(n_states, n_obs)
    backward[:, n_obs] = ones(n_states)

    for t in n_obs-1:-1:1
        for i in 1:n_states
            backward[i, t] = sum(transition_matrix[i, :] .* emission_matrix[:, observations[t+1]] .* backward[:, t+1])
        end
    end

    # Compute the smoothed probabilities
    smoothed_probs = forward .* backward
    smoothed_probs ./= sum(smoothed_probs, dims=1)

    return smoothed_probs
end


# Our state space is defined by a hyperrectangle. It is spanned the origin and [N_a, N_b, N_c, ...].
# The initial probability is a distribution on this hyperrectangle.
# The forward probabilities are computed from the initial probability and the "rate matrix"

function propagate_probs(p::AbstractArray, reactions::AbstractJumpSet)
    
end
