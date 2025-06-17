
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def quantize_per_tensor_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, common scale and zero_point
    input1 = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scale1 = 0.5
    zero_point1 = 10
    dtype1 = torch.int8

    input_dict1 = {
        "input": input1,
        "scale": scale1,
        "zero_point": zero_point1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))


    return list_of_inputs

generated_inputs["torch.quantize_per_tensor"] = quantize_per_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.quantize_per_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantize_per_tensor'.")

check_valid('torch.quantize_per_tensor', generated_inputs['torch.quantize_per_tensor'], lib="torch")
