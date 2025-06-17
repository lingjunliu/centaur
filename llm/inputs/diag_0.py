
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def diag_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor (vector) with diagonal=0
    input_1 = torch.randn(3).numpy()
    input_dict_1 = {"input": input_1, "diagonal": 0, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 1D tensor (vector) with diagonal=1
    input_2 = torch.randn(3).numpy()
    input_dict_2 = {"input": input_2, "diagonal": 1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 1D tensor (vector) with diagonal=-1
    input_3 = torch.randn(3).numpy()
    input_dict_3 = {"input": input_3, "diagonal": -1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 2D tensor (matrix) with diagonal=0
    input_4 = torch.randn(3, 3).numpy()
    input_dict_4 = {"input": input_4, "diagonal": 0, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 2D tensor (matrix) with diagonal=1
    input_5 = torch.randn(3, 3).numpy()
    input_dict_5 = {"input": input_5, "diagonal": 1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.diag"] = diag_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diag'.")

check_valid('torch.diag', generated_inputs['torch.diag'], lib="torch")
