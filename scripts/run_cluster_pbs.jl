
using Distributed

machines = readlines(ENV["PBS_NODEFILE"])
jobid = ENV["PBS_JOBID"]
addprocs(machines; exename=`/opt/pbs/bin/pbs_attach -j $jobid $(Sys.BINDIR)/julia`)

include(ARGS[2])
