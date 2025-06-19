
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def softshrink_inputs():
    list_of_inputs = []

    input_np = np.array([-1, -0.5, 0, 0.5, 1], dtype=np.float32)
    lambd = 0.5
    input_dict = {"input": input_np, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    lambd = 1.0
    input_dict = {"input": input_np, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([[-1, -0.5], [0, 0.5], [1, 2]], dtype=np.float32)
    lambd = 0.25
    input_dict = {"input": input_np, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([[[1, -1], [0.5, -0.5]], [[0, 0], [1, -2]]], dtype=np.float64)
    lambd = 0.75
    input_dict = {"input": input_np, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([[-1.5, -1], [-0.5, 0], [0.5, 1], [1.5, 2]], dtype=np.float32)
    lambd = 1.2
    input_dict = {"input": input_np, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([-0.2, -0.1, 0, 0.1, 0.2], dtype=np.float32)
    lambd = 0.3
    input_dict = {"input": input_np, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.softshrink"] = softshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.softshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softshrink'.")

check_valid('torch.nn.functional.softshrink', generated_inputs['torch.nn.functional.softshrink'], lib="torch")
