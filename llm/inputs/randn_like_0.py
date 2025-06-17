
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def randn_like_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.random.randn(2, 3).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor
    input2 = np.random.randint(-5, 5, size=(4, 4), dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.randn_like"] = randn_like_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.randn_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.randn_like'.")

check_valid('torch.randn_like', generated_inputs['torch.randn_like'], lib="torch")
