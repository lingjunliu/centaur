
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_median_1_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input_1 = torch.randn(5).numpy()
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int tensor
    input_2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D float tensor with negative values
    input_3 = torch.randn(2, 3, 5) * -1.0
    input_3 = input_3.numpy()
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: 4D float tensor
    input_4 = torch.randn(2, 2, 2, 2).numpy()
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1D int tensor with large values
    input_5 = torch.randint(100, 200, (10,)).numpy()
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    return list_of_inputs

generated_inputs["torch.median_1"] = torch_median_1_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.median_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.median_1'.")

check_valid('torch.median', generated_inputs['torch.median_1'], lib="torch")
