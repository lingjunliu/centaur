
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bessel_y1_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative float tensor
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict2 = {"input": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Zero value (avoiding zero for bessel_y1)
    input3 = np.array([0.001], dtype=np.float32)
    input_dict3 = {"input": input3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger float values
    input4 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    input_dict4 = {"input": input4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multi-dimensional float tensor
    input5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict5 = {"input": input5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.special.bessel_y1"] = bessel_y1_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.bessel_y1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.bessel_y1'.")

check_valid('torch.special.bessel_y1', generated_inputs['torch.special.bessel_y1'], lib="torch")
