
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_any_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean tensor
    input_1 = torch.tensor([True, False, True]).numpy()
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer tensor with values that evaluate to True/False
    input_2 = torch.tensor([0, 1, 0, -1, 2]).numpy()
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Float tensor
    input_3 = torch.tensor([0.0, 1.0, -1.5, 0.0]).numpy()
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.any_1"] = torch_any_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.any_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.any_1'.")

check_valid('torch.any', generated_inputs['torch.any_1'], lib="torch")
