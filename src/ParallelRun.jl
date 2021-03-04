using Distributed

function reduce_results(res1, results...)
    new_res = typeof(res1)()
    for k in keys(res1)
        new_res[k] = vcat(res1[k], (r[k] for r in results)...)
    end
    new_res
end

function run_parallel(system, algorithm, num_responses)
    batches = Int[]

    batch_size = clamp(floor(num_responses / nworkers()), 1, 10)

    while num_responses > 0
        batch = min(num_responses, batch_size)
        push!(batches, batch)
        num_responses -= batch
    end

    result = pmap(batch -> mutual_information(system, algorithm; num_responses=batch), batches)
    vcat(result...)
end
