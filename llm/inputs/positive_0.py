
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def positive_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor
    input2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor with negative values
    input3 = (torch.randn(2, 3, 4) * 10 - 5).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 4D complex tensor
    input4 = (torch.randn(2, 2, 2, 2, dtype=torch.complex64) + 1j * torch.randn(2, 2, 2, 2)).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar float tensor
    input5 = torch.tensor(3.14).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = torch.empty(0).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with a single negative value
    input7 = torch.ones(2, 2)
    input7[0, 0] = -1
    input7 = input7.numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.positive"] = positive_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.positive' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.positive'.")

check_valid('torch.positive', generated_inputs['torch.positive'], lib="torch")
