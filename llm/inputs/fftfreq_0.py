
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def fftfreq_inputs():
    list_of_inputs = []

    # Test case 1: Basic test with only n
    input_dict = {
        "n": np.int32(5),
        "d": np.float64(1.0),
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: With n and d
    input_dict = {
        "n": np.int32(10),
        "d": np.float64(0.1),
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: With n, d, and dtype
    input_dict = {
        "n": np.int32(7),
        "d": np.float32(0.5),
        "out": None,
        "dtype": torch.float64,
        "layout": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 4: n is even
    input_dict = {
        "n": np.int32(8),
        "d": np.float64(1.0),
        "out": None,
        "dtype": None,
        "layout": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: With out tensor
    input_dict = {
        "n": np.int32(6),
        "d": np.float32(1.0),
        "out": None,
        "dtype": torch.float32,
        "layout": None,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.fftfreq"] = fftfreq_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.fftfreq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fftfreq'.")

check_valid('torch.fft.fftfreq', generated_inputs['torch.fft.fftfreq'], lib="torch")
