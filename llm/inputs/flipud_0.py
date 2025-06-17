
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def flipud_inputs():
    list_of_inputs = []

    # Input 1: 1D integer tensor
    input1 = torch.arange(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with negative values
    input3 = (torch.randn(2, 3, 2) * -1).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 4D tensor
    input4 = torch.randn(1, 2, 3, 4).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D integer tensor with specific values
    input5 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D complex tensor
    input6 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 2D bool tensor
    input7 = torch.tensor([[True, False], [False, True]]).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.flipud"] = flipud_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.flipud' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flipud'.")

check_valid('torch.flipud', generated_inputs['torch.flipud'], lib="torch")
