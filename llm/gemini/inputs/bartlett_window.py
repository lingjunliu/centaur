
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bartlett_window_inputs():
    list_of_inputs = []

    input_dict = {
        "window_length": np.int32(5),
        "periodic": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": np.int64(10),
        "periodic": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": np.int32(7),
        "periodic": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "window_length": np.int64(12),
        "periodic": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "window_length": np.int32(3),
        "periodic": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "window_length": np.int64(8),
        "periodic": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = bartlett_window_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bartlett_window', generated_inputs)
