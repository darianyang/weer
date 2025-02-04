#!/bin/bash

# Make sure environment is set
#source env.sh

# Clean up
#rm -f west.log

# Run w_run
#w_init --bstate 'basis,4,4,4,4,4,4' --tstate 'target,1,1,1,1,1,1'

# this is good: label,prob,auxref(state index)
#w_init --bstate 'basis,1,1361' 

w_run > west.log

#w_run --work-manager processes "$@" &> west.log
