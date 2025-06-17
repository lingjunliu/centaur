
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def polygamma_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    n = 0
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"n": n, "input": input_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor
    n = 1
    input_tensor = torch.randint(1, 10, (2, 2)).numpy()
    input_dict = {"n": n, "input": input_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Higher order polygamma, different shape
    n = 2
    input_tensor = torch.randn(5).numpy()
    input_dict = {"n": n, "input": input_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values in input
    n = 0
    input_tensor = torch.randn(2, 3) * -1.0
    input_tensor = input_tensor.numpy()
    input_dict = {"n": n, "input": input_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger tensor
    n = 1
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"n": n, "input": input_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: n is a larger integer
    n = 5
    input_tensor = torch.randn(3, 3).numpy()
    input_dict = {"n": n, "input": input_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.polygamma"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.polygamma'.")

check_valid('torch.polygamma', generated_inputs['torch.polygamma'], lib="torch")
