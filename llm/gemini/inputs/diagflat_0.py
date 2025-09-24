
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def diagflat_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor (vector) with offset 0
    a = torch.randn(3).numpy()
    input_dict = {"input": a, "offset": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor (vector) with offset 1
    a = torch.randn(4).numpy()
    input_dict = {"input": a, "offset": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor (vector) with offset -1
    a = torch.randn(5).numpy()
    input_dict = {"input": a, "offset": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor with offset 0
    a = torch.randn(2, 2).numpy()
    input_dict = {"input": a, "offset": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor with offset 0
    a = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": a, "offset": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D integer tensor with offset 0
    a = torch.randint(0, 10, (3,)).numpy()
    input_dict = {"input": a, "offset": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D tensor (vector) with negative values and offset 2
    a = torch.randn(3) * -1
    a = a.numpy()
    input_dict = {"input": a, "offset": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor (vector) with negative values and offset -2
    a = torch.randn(3) * -1
    a = a.numpy()
    input_dict = {"input": a, "offset": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.diagflat"] = diagflat_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.diagflat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diagflat'.")

check_valid('torch.diagflat', generated_inputs['torch.diagflat'], lib="torch")
