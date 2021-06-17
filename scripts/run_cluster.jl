
using MPIClusterManagers
using Distributed

nprocs = length(readlines(ENV["PBS_NODEFILE"]))
manager = MPIManager(np=nprocs, master_tcp_interface=gethostname())

addprocs(manager)
# addprocs(manager, exename=`$(Sys.BINDIR)/julia -J$(pwd())/GMcmcSysimage.so`)

using Logging
@info "Successfully launched MPI processes"

include(ARGS[2])
