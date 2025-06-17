
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def var_mean_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 4, 5)
    input_dict1 = {
        "input": input1,
        "dim": (0, 1),
        "unbiased": True,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict2 = {
        "input": input2,
        "dim": (0,),
        "unbiased": False,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randint(0, 10, size=(2, 3)).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "dim": (1,),
        "unbiased": True,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    input_dict4 = {
        "input": input4,
        "dim": None,
        "unbiased": True,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.var_mean_1"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_1'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_1'], lib="torch")
