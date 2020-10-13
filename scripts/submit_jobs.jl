using DrWatson

my_args = Dict(
    "algorithm" => "thermodynamic_integration",
    "duration" => collect(range(50.0, 500.0, length=10)),
    "N" => Vector(1:8),
    "num_responses" => 2
)
res = tmpsave(dict_list(my_args))

print(res)
