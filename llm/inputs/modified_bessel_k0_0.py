
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def modified_bessel_k0_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Another float tensor with different values and shape
    input2 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Integer tensor (should be cast to float internally)
    input3 = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with zero
    input4 = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multi-dimensional tensor
    input5 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.special.modified_bessel_k0"] = modified_bessel_k0_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.modified_bessel_k0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.modified_bessel_k0'.")

check_valid('torch.special.modified_bessel_k0', generated_inputs['torch.special.modified_bessel_k0'], lib="torch")
