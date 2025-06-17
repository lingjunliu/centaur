
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def msort_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 3: 3D float tensor with negative values. Reshape to 2D
    input3 = torch.randn(2, 3, 2) * 10 - 5
    input3 = input3.numpy()
    input3 = input3.reshape(2, -1)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D tensor with different data types
    input4 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor with zeros
    input5 = torch.zeros(3, 3).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 1D Float Tensor - Reshaping to be 2D
    input6 = torch.randn(5).numpy()
    input6 = input6.reshape(1,-1)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 2D int Tensor
    input7 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.msort"] = msort_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.msort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.msort'.")

check_valid('torch.msort', generated_inputs['torch.msort'], lib="torch")
