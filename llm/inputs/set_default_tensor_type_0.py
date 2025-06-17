
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def set_default_tensor_type_inputs():
    list_of_inputs = []

    input_dict = {
        "t": "torch.FloatTensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "t": "torch.DoubleTensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "t": "torch.HalfTensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "t": "torch.BFloat16Tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.set_default_tensor_type"] = set_default_tensor_type_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_default_tensor_type' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_default_tensor_type'.")

check_valid('torch.set_default_tensor_type', generated_inputs['torch.set_default_tensor_type'], lib="torch")
