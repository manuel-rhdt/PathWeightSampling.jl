using Distributed
using Logging
using Printf

function reduce_results(res1, results...)
    new_res = typeof(res1)()
    for k in keys(res1)
        new_res[k] = vcat(res1[k], (r[k] for r in results)...)
    end
    new_res
end

const MAX_BATCH_SIZE = 4

function run_parallel(systemfn, algorithm, num_responses)
    batches = Int[]

    N = num_responses
    batch_size = clamp(floor(num_responses / nworkers()), 1, MAX_BATCH_SIZE)

    while num_responses > 0
        batch = min(num_responses, batch_size)
        push!(batches, batch)
        num_responses -= batch
    end

    @everywhere begin
        system = $systemfn()
        global compiled_system = GaussianMcmc.compile(system)
    end

    channel = RemoteChannel(()->Channel())
    @info "Starting Parallel computation"
    result = @sync begin
        # display progress
        progress = 0
        @async while true
            val = take!(channel)
            if val === nothing
                break
            end
            (hostname, elapsed_time, batch_size) = val
            progress += batch_size
            percent_done = @sprintf "%6.2f %%" (progress / N * 100)
            @info "Finished batch" hostname elapsed_time batch_size percent_done
        end

        @sync begin
            result = pmap(batches) do batch
                time_stats = @timed result = _mi_inner(Main.compiled_system, algorithm, batch, false)
                elapsed_time = time_stats.time
                hostname = gethostname()
                put!(channel, (hostname, elapsed_time, batch))
                yield()
                result
            end
            put!(channel, nothing)
            vcat(result...)
        end
    end
    @info "Finished Parallel computation"
    result
end

