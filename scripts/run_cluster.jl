
using MPIClusterManagers
using Logging
using Distributed

nprocs = length(readlines(ENV["PBS_NODEFILE"]))
manager = MPIManager(np=nprocs, master_tcp_interface=gethostname())
addprocs(manager)

@info "Successfully launched MPI processes"

include(ARGS[2])
