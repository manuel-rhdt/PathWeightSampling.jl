
N = 0
if haskey(ENV, "PBS_ARRAYID")
    global N = parse(Int, ENV["PBS_ARRAYID"])
end

using MPIClusterManagers
using Logging
using Distributed

manager = MPIManager(master_tcp_interface=gethostname())
addprocs(manager)

@info "Successfully launched MPI processes"

include("simple_network.jl")