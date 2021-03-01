using Distributed

function reduce_results(res1, results...)
    new_res = typeof(res1)()
    for k in keys(res1)
        new_res[k] = vcat(res1[k], (r[k] for r in results)...)
    end
    new_res
end

function run_parallel(system, algorithm, num_responses)
    result = pmap(x -> mutual_information(system, algorithm; num_responses=1), 1:num_responses)
    vcat(result...)
end
