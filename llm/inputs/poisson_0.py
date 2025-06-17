
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def poisson_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D tensor with float rates
    rates1 = np.random.rand(2, 3) * 5
    input_dict1 = {"input": rates1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with integer rates
    rates2 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    input_dict2 = {"input": rates2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor
    rates3 = np.random.rand(2, 2, 2) * 10
    input_dict3 = {"input": rates3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Single element tensor
    rates4 = np.array([3.5], dtype=np.float32)
    input_dict4 = {"input": rates4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger tensor
    rates5 = np.random.rand(5, 5) * 2
    input_dict5 = {"input": rates5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Integer type rates
    rates6 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict6 = {"input": rates6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.poisson"] = poisson_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.poisson' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.poisson'.")

check_valid('torch.poisson', generated_inputs['torch.poisson'], lib="torch")
