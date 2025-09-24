
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_clamp_inputs():
    list_of_inputs = []

    # Case 1: Basic clamping with float min and max
    input_tensor = torch.randn(4).float().numpy()
    min_val = -0.5
    max_val = 0.5
    out_tensor = torch.empty(4).float().numpy()

    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Clamping with no lower bound (min=None)
    input_tensor = torch.randn(2, 3).float().numpy()
    min_val = None
    max_val = 0.75
    out_tensor = torch.empty(2, 3).float().numpy()

    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Clamping with no upper bound (max=None)
    input_tensor = torch.randn(3, 1).float().numpy()
    min_val = -1.2
    max_val = None
    out_tensor = torch.empty(3, 1).float().numpy()

    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Clamping with min > max
    input_tensor = torch.randn(5).float().numpy()
    min_val = 1.0
    max_val = 0.0
    out_tensor = torch.empty(5).float().numpy()

    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Clamping with a scalar input tensor
    input_tensor = torch.tensor(2.5).float().numpy()
    min_val = 1.0
    max_val = 3.0
    out_tensor = np.array(0.0, dtype=np.float32)

    input_dict = {
        "input": input_tensor,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.clamp_3"] = torch_clamp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clamp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_3'.")

check_valid('torch.clamp', generated_inputs['torch.clamp_3'], lib="torch")
