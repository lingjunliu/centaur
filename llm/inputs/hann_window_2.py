
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hann_window_inputs():
    list_of_inputs = []

    input_dict = {
        "window_length": np.int32(5),
        "periodic": True,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": np.int64(10),
        "periodic": False,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": np.int32(7),
        "periodic": True,
        "dtype": torch.float16,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": np.int64(12),
        "periodic": False,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": np.int32(3),
        "periodic": True,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.hann_window_2"] = hann_window_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hann_window_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hann_window_2'.")

check_valid('torch.hann_window', generated_inputs['torch.hann_window_2'], lib="torch")
