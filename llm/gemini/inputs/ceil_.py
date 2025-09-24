
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def ceil__inputs():
    list_of_inputs = []

    input_dict = {"input": np.array([1.2, 2.5, -3.1, -0.5])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([[1.2, 2.5], [-3.1, -0.5]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([1, 2, 3], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([0.0, -0.0, 1e-8, -1e-8])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([[[1.2, 2.5], [-3.1, -0.5]], [[-1.2, -2.5], [3.1, 0.5]]])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([np.nan, np.inf, -np.inf, 1.5])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = ceil__inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ceil_', generated_inputs)
