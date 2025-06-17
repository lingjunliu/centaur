
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import copy

def remove_weight_norm_inputs():
    list_of_inputs = []

    # Example 1: Linear layer
    linear_module = nn.Linear(10, 20)
    nn_utils.weight_norm(linear_module, name="weight")
    
    input_dict = {
        "module": linear_module,
        "name": "weight"
    }
    
    linear_module_no_wn = copy.deepcopy(linear_module)
    nn_utils.remove_weight_norm(linear_module_no_wn, name="weight")
    input_dict["module"] = linear_module_no_wn
    
    list_of_inputs.append(input_dict)
    

    # Example 2: Conv2d layer
    conv2d_module = nn.Conv2d(3, 16, kernel_size=3)
    nn_utils.weight_norm(conv2d_module, name="weight")
    input_dict = {
        "module": conv2d_module,
        "name": "weight"
    }
    conv2d_module_no_wn = copy.deepcopy(conv2d_module)
    nn_utils.remove_weight_norm(conv2d_module_no_wn, name="weight")
    input_dict["module"] = conv2d_module_no_wn
    
    list_of_inputs.append(input_dict)

    # Example 3: ConvTranspose2d layer
    convtranspose2d_module = nn.ConvTranspose2d(3, 16, kernel_size=3)
    nn_utils.weight_norm(convtranspose2d_module, name="weight")
    input_dict = {
        "module": convtranspose2d_module,
        "name": "weight"
    }
    convtranspose2d_module_no_wn = copy.deepcopy(convtranspose2d_module)
    nn_utils.remove_weight_norm(convtranspose2d_module_no_wn, name="weight")
    input_dict["module"] = convtranspose2d_module_no_wn
    
    list_of_inputs.append(input_dict)
    
    # Example 4: Different name of weight parameter
    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.custom_weight = nn.Parameter(torch.randn(10, 20))
        
        def forward(self, x):
            return x @ self.custom_weight

    custom_module = CustomModule()
    nn_utils.weight_norm(custom_module, name="custom_weight")
    input_dict = {
        "module": custom_module,
        "name": "custom_weight"
    }
    custom_module_no_wn = copy.deepcopy(custom_module)
    nn_utils.remove_weight_norm(custom_module_no_wn, name="custom_weight")
    input_dict["module"] = custom_module_no_wn
    
    list_of_inputs.append(input_dict)

    # Example 5: Conv1d Layer
    conv1d_module = nn.Conv1d(3, 16, kernel_size=3)
    nn_utils.weight_norm(conv1d_module, name="weight")

    input_dict = {
        "module": conv1d_module,
        "name": "weight"
    }
    conv1d_module_no_wn = copy.deepcopy(conv1d_module)
    nn_utils.remove_weight_norm(conv1d_module_no_wn, name="weight")
    input_dict["module"] = conv1d_module_no_wn
    
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["torch.nn.utils.remove_weight_norm"] = remove_weight_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.remove_weight_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.remove_weight_norm'.")

check_valid('torch.nn.utils.remove_weight_norm', generated_inputs['torch.nn.utils.remove_weight_norm'], lib="torch")
