
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def ZeroPad1d_inputs():
    list_of_inputs = []

    # Test case 1: Integer padding
    input_dict = {"padding": 2, "input": np.random.randn(1, 2, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Tuple padding (different left and right)
    input_dict = {"padding": (3, 1), "input": np.random.randn(1, 2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Zero padding
    input_dict = {"padding": 0, "input": np.random.randn(1, 2, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Large padding values
    input_dict = {"padding": (10, 5), "input": np.random.randn(1, 2, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Same padding on both sides using a tuple
    input_dict = {"padding": (4, 4), "input": np.random.randn(1, 2, 6)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = ZeroPad1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ZeroPad1d', generated_inputs)
