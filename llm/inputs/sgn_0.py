
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sgn_inputs():
    list_of_inputs = []

    # Example 1: Float tensor
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Int tensor
    input_tensor = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Complex tensor
    input_tensor = torch.complex(torch.randn(2, 3), torch.randn(2, 3)).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sgn"] = sgn_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sgn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sgn'.")

check_valid('torch.sgn', generated_inputs['torch.sgn'], lib="torch")
