
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: 1D int tensor
    input1 = np.array([0, 1, -1, 2, -2, 127, -128], dtype=np.int8)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.bitwise_not"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_not'.")

check_valid('torch.bitwise_not', generated_inputs['torch.bitwise_not'], lib="torch")
