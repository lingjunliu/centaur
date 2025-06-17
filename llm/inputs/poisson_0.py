
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def poisson_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor with float rates
    rates1 = np.random.rand(2, 3) * 5
    input_dict1 = {"input": rates1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with integer rates
    rates2 = np.random.randint(1, 6, size=5).astype(np.float32)
    input_dict2 = {"input": rates2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with different rates
    rates3 = np.random.rand(1, 2, 2) * 10
    input_dict3 = {"input": rates3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Single value tensor
    rates4 = np.array(3.5)
    input_dict4 = {"input": rates4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Another 2D tensor with integer rates, large values
    rates5 = np.random.randint(10, 20, size=(3, 4)).astype(np.float64)
    input_dict5 = {"input": rates5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Use generator (remove generator input for now, handle separately)
    #generator = torch.Generator()
    #rates6 = np.random.rand(2, 2) * 3
    #input_dict6 = {"input": rates6, "generator": generator}
    #list_of_inputs.append(copy.deepcopy(input_dict6))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.poisson"] = poisson_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.poisson' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.poisson'.")

check_valid('torch.poisson', generated_inputs['torch.poisson'], lib="torch")
