using Distributed
using Logging

function reduce_results(res1, results...)
    new_res = typeof(res1)()
    for k in keys(res1)
        new_res[k] = vcat(res1[k], (r[k] for r in results)...)
    end
    new_res
end

function run_parallel(systemfn, algorithm, num_responses; progress=true)
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

    p = Progress(length(batches); enabled=progress)
    result = progress_pmap(batches, progress=p) do batch
        time_stats = @timed result = _mi_inner(Main.compiled_system, algorithm, batch, false)
        elapsed_time = time_stats.time
        hostname = gethostname()
        batch_size = batch
        @info "Finished batch" hostname elapsed_time batch_size
        result
    end
    vcat(result...)
end

