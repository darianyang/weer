#!/bin/bash

# Make sure environment is set
#source env.sh

# Clean up
rm -f west.log

# Run w_run
#w_init --bstate 'basis,4,4,4,4,4,4' --tstate 'target,1,1,1,1,1,1'
# should be: label,prob,auxref(stateindex)
w_init --bstate 'basis,1,800' --segs-per-state 6
#w_init --bstate 'basis,1,800' --verbose
w_run "$@" > west.log

#w_run --work-manager processes "$@" &> west.log
#w_run --parallel > west.log
#w_run --work-manager threads > west.log
#w_run --work-manager processes > west.log
