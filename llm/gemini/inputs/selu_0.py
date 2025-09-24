
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def selu_inputs():
    list_of_inputs = []

    input_1 = np.random.randn(3, 4).astype(np.float32)
    input_dict_1 = {
        "input": input_1,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict_2 = {
        "input": input_2,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = np.array([-1, 0, 1, -2, 2], dtype=np.float32)
    input_dict_3 = {
        "input": input_3,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    input_5 = np.random.rand(2, 2, 2).astype(np.float32)
    input_dict_5 = {
        "input": input_5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.nn.functional.selu"] = selu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.selu'.")

check_valid('torch.nn.functional.selu', generated_inputs['torch.nn.functional.selu'], lib="torch")
