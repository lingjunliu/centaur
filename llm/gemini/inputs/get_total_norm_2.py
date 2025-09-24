
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def get_total_norm_inputs():
    list_of_inputs = []

    # Input 1: List of float tensors, norm_type=2
    tensor_list = [torch.randn(3, 4).float().numpy(), torch.randn(5, 2).float().numpy()]
    norm_type = 2
    input_dict = {"parameters": tensor_list, "norm_type": norm_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of float tensors, norm_type=1
    tensor_list = [torch.randn(2, 2).float().numpy(), torch.randn(3, 3).float().numpy()]
    norm_type = 1
    input_dict = {"parameters": tensor_list, "norm_type": norm_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of tensors with different shapes and types, norm_type=np.inf
    tensor_list = [torch.randn(1, 5).float().numpy(), torch.randn(2,).float().numpy(), torch.randn(3, 1, 2).double().numpy()]
    norm_type = np.inf
    input_dict = {"parameters": tensor_list, "norm_type": norm_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of tensors with negative values, norm_type=3
    tensor_list = [torch.randn(2, 3).float().numpy(), torch.randn(4, 1).float().numpy()]
    norm_type = 3
    input_dict = {"parameters": tensor_list, "norm_type": norm_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single tensor in a list, norm_type=0.5
    tensor_list = [torch.randn(5, 5).float().numpy()]
    norm_type = 0.5
    input_dict = {"parameters": tensor_list, "norm_type": norm_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of float tensors with a norm_type = 1.5
    tensor_list = [torch.randn(2, 2).float().numpy(), torch.randn(3, 3).float().numpy()]
    norm_type = 1.5
    input_dict = {"parameters": tensor_list, "norm_type": norm_type}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.utils.get_total_norm_2"] = get_total_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.get_total_norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.get_total_norm_2'.")

check_valid('torch.nn.utils.get_total_norm', generated_inputs['torch.nn.utils.get_total_norm_2'], lib="torch")
