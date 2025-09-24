
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def linear_inputs():
    list_of_inputs = []

    # Case 1: Basic case with bias
    input_np = np.random.randn(3, 4).astype(np.float32)
    weight_np = np.random.randn(5, 4).astype(np.float32)
    bias_np = np.random.randn(5).astype(np.float32)
    input_dict = {"input": input_np, "weight": weight_np, "bias": bias_np}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: No bias
    input_np = np.random.randn(2, 6).astype(np.float64)
    weight_np = np.random.randn(7, 6).astype(np.float64)
    input_dict = {"input": input_np, "weight": weight_np, "bias": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Batch input
    input_np = np.random.randn(2, 3, 4).astype(np.float32)
    weight_np = np.random.randn(5, 4).astype(np.float32)
    bias_np = np.random.randn(5).astype(np.float32)
    input_dict = {"input": input_np, "weight": weight_np, "bias": bias_np}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values
    input_np = np.random.randn(2, 3) * -1.0
    weight_np = np.random.randn(4, 3) * -1.0
    bias_np = np.random.randn(4) * -1.0
    input_dict = {"input": input_np, "weight": weight_np, "bias": bias_np}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.linear"] = linear_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.linear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.linear'.")

check_valid('torch.nn.functional.linear', generated_inputs['torch.nn.functional.linear'], lib="torch")
