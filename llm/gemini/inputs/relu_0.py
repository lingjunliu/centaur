
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def relu_inputs():
    list_of_inputs = []

    input_float = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input_float,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_int = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    input_dict = {
        "input": input_int,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_neg = torch.randn(2, 3, 4).numpy() - 1
    input_dict = {
        "input": input_neg,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_zeros = np.zeros((5, 5), dtype=np.float32)
    input_dict = {
        "input": input_zeros,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_3d = torch.randn(1, 5, 5, 5).numpy()
    input_dict = {
        "input": input_3d,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.relu"] = relu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.relu'.")

check_valid('torch.nn.functional.relu', generated_inputs['torch.nn.functional.relu'], lib="torch")
