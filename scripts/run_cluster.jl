
using MPIClusterManagers
using Logging
using Distributed

manager = MPIManager(master_tcp_interface=gethostname())
addprocs(manager)

@assert nworkers() >= 2
@info "Successfully launched MPI processes"

include(ARGS[2])