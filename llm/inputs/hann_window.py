
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def hann_window_inputs():
    generated_inputs = []

    input_dict = {
        "window_length": 5,
        "periodic": False,
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 10,
        "periodic": True,
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": True
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 7,
        "periodic": False,
        "dtype": torch.float16,
        "layout": "strided",
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 12,
        "periodic": True,
        "dtype": torch.bfloat16,
        "layout": "strided",
        "requires_grad": True
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 3,
        "periodic": False,
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 15,
        "periodic": True,
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": True
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    
    return generated_inputs

generated_inputs = hann_window_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('hann_window', generated_inputs)
