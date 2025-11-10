#!/bin/bash

# 1. Get the 'ran' sum
ran=$(find eval/patched_drivers/ -mindepth 2 -maxdepth 2 -name log.txt -exec tail -n 1 -q {} + | awk '{sum += $NF} END {print sum}')
                          
# 2. Get the 'total' count
total=$(find eval/patched_drivers/ -mindepth 2 -maxdepth 2 -name "*.pkl" | wc -l)
                                             
# 3. Calculate and print percentage using awk
#    We pass the shell variables into awk using the -v flag.
awk -v r="$ran" -v t="$total" 'BEGIN {
    if (t > 0) {
        printf "Coverage completed: %.2f%% (%d/%d)\n", (r / t) * 100, r, t
    } else {
        print "Coverage completed: 0.00%"
    }              
}'