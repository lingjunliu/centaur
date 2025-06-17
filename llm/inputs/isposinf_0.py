
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def isposinf_inputs():
    list_of_inputs = []

    a = np.array([-np.inf, np.inf, 1.2])
    input_dict = {"input": torch.tensor(a, dtype=torch.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([float('inf'), float('inf'), float('inf')])
    input_dict = {"input": torch.tensor(a, dtype=torch.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[float('inf'), 2.0], [3.0, float('inf')]])
    input_dict = {"input": torch.tensor(a, dtype=torch.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[float('inf'), 2.0], [3.0, float('inf')]], [[1.0, -float('inf')], [float('inf'), 4.0]]])
    input_dict = {"input": torch.tensor(a, dtype=torch.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([1.0, 2.0, 3.0])  
    input_dict = {"input": torch.tensor(a, dtype=torch.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.isposinf"] = isposinf_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isposinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isposinf'.")

check_valid('torch.isposinf', generated_inputs['torch.isposinf'], lib="torch")
