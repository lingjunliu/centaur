
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def prepare_multiprocessing_environment_inputs():
    list_of_inputs = []

    # Input 1: rank = 0
    input_dict = {
        "rank": 0,
        "input": np.array([0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: rank = 1
    input_dict = {
        "rank": 1,
        "input": np.array([1])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: rank = 2
    input_dict = {
        "rank": 2,
        "input": np.array([2])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: rank = 10
    input_dict = {
        "rank": 10,
        "input": np.array([10])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: rank = 100
    input_dict = {
        "rank": 100,
        "input": np.array([100])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = prepare_multiprocessing_environment_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('prepare_multiprocessing_environment', generated_inputs)
