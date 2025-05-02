import time
import numpy as np
import traceback
import os
import pickle
import sys

from .ea import Configuration, Mutator, optimize
from .definitions import map_defs, get_definition
from .input_generators import get_random_input
from utils.api_utils import get_driver
from utils.misc import create_subdir, get_tmp_dir

# [To-do] 
# def create_z3_args(..):
#     ..

# [To-do] 
# def collect_constraints(..):
#     ..

# [To-do] 
# def solve_constraints(..):
#     ..

def run_api_with_duration(api, duration, n_max=0, limit=30, print_details=False):
    driver = get_driver(api)

    print(f"Optimizing for {api} with a {duration} second budget")
    execution_time = 0
    start = time.time()
    elapsed = 0
    valid = 0
    invalid = 0
    seed = 200
    generated_inputs = []
    definition = get_definition(api, z3=True)
    if len(definition["ruleset"]) == 0:
        print(f"No invariants learned for {api}")
        return

    # [To-do]

if __name__ == "__main__":
    # Run scatter for 30 minutes
    duration = 30 # seconds
    limit = 10  # random restart after <limit> seconds
    print_details = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else False
    
    run_api_with_duration("scatter", duration, print_details=print_details, limit=limit)
    
    # Run atan2 for 30 seconds
    # run_api_with_duration("atan2", duration, print_details=print_details, limit=limit)
    
    # Run argmin for 30 seconds
    # run_api_with_duration("argmin", duration, print_details=print_details, limit=limit)
    
    # Run conv_transpose2d for 30 seconds
    # run_api_with_duration("conv_transpose2d", duration, print_details=print_details, limit=limit)
