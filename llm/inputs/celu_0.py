
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def celu_inputs():
    list_of_inputs = []

    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    alpha1 = 1.0
    input_dict1 = {"input": input1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-1.0, -0.5], [0.0, 0.5]], dtype=np.float64)
    alpha2 = 0.5
    input_dict2 = {"input": input2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-2, -1], [0, 1]], dtype=np.float32)
    alpha3 = 2.0
    input_dict3 = {"input": input3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 2, 3], dtype=np.float32)
    alpha4 = 0.75
    input_dict4 = {"input": input4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    alpha5 = 1.5
    input_dict5 = {"input": input5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.celu'.")

check_valid('torch.celu', generated_inputs['torch.celu'], lib="torch")
