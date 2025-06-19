
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lazy_instance_norm1d_inputs():
    list_of_inputs = []

    input = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "input": input,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 5, 6).astype(np.float64)
    input_dict = {
        "input": input,
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(4, 2, 8).astype(np.float16)
    input_dict = {
        "input": input,
        "eps": 1e-06,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LazyInstanceNorm1d"] = lazy_instance_norm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LazyInstanceNorm1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LazyInstanceNorm1d'.")

check_valid('torch.nn.LazyInstanceNorm1d', generated_inputs['torch.nn.LazyInstanceNorm1d'], lib="torch")
