using Distributed

function reduce_results(res1, results...)
    new_res = typeof(res1)()
    for k in keys(res1)
        new_res[k] = vcat(res1[k], (r[k] for r in results)...)
    end
    new_res
end

function run_parallel(systemfn, algorithm, num_responses)
    batches = Int[]

    batch_size = clamp(floor(num_responses / nworkers()), 1, 10)

    while num_responses > 0
        batch = min(num_responses, batch_size)
        push!(batches, batch)
        num_responses -= batch
    end

    @everywhere begin
        system = $systemfn()
        global compiled_system = GaussianMcmc.compile(system)
    end

    result = pmap(batch -> _mi_inner(Main.compiled_system, algorithm, batch), batches)
    vcat(result...)
end

