
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def torch_jit_Error_inputs():
    list_of_inputs = []

    input_dict = {
        "msg": "This is a generic error message."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Division by zero encountered."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Index out of bounds."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Type mismatch: Expected int but got float."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Shape mismatch: Input tensors must have compatible shapes."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "CUDA error: device-side assert triggered"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Value out of range."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.jit.Error"] = torch_jit_Error_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.Error' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.Error'.")

check_valid('torch.jit.Error', generated_inputs['torch.jit.Error'], lib="torch")
