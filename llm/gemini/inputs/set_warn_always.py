
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def set_warn_always_inputs():
    list_of_inputs = []

    input_dict = {
        "warn_always": np.array(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "warn_always": np.array(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "warn_always": np.array([True])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "warn_always": np.array([False])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "warn_always": np.array([[True]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "warn_always": np.array([[False]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "warn_always": np.array([True, False, True])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "warn_always": np.array([[True, False], [False, True]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = set_warn_always_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_warn_always', generated_inputs)
