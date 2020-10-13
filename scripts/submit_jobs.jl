using DrWatson
import JSON
using Random

my_args = Dict(
    "algorithm" => "thermodynamic_integration",
    "duration" => collect(range(50.0, 500.0, length=10)),
    "N" => collect(1:8),
    "num_responses" => 1000
)

function runsave(dicts, tmp=projectdir("_research", "tmp"), prefix="", suffix="json", l=8)
    mkpath(tmp)
    n = length(dicts)
    indices = map(string, 1:n)
    existing = readdir(tmp)
    filename = prefix * randstring(l)
    r = filename .* "." .* indices .* "." .* suffix
    while !isdisjoint(existing, r)
        filename = prefix * randstring(l)
        r = filename .* "." .* indices .* "." .* suffix
    end

    for (i, path) ∈ enumerate(r)
        open(joinpath(tmp, path), "w") do io
            JSON.print(io, copy(dicts[i]))
        end
    end

    filename
end

dicts = dict_list(my_args)
filename = runsave(dicts)

jobscript = """
    export JULIA_PROJECT=$(projectdir())

    julia -e "using InteractiveUtils; versioninfo()"
    julia $(projectdir("scripts", "simple_network.jl")) $filename.\$PBS_ARRAYID.json
    """

result = ""

out_dir = projectdir("data", "output")
mkpath(out_dir)

open(`qsub -N TI1 -l nodes=1:ppn=1:highcore,mem=4gb,walltime=24:00:00 -t 1-$(length(dicts)) -j oe -o $out_dir`, "r+") do io
    print(io, jobscript)
    close(io.in)
    global result *= read(io, String)
end

print(result)
