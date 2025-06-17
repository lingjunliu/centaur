
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ifftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor, specify dim
    input2 = torch.randint(-5, 5, (4, 6)).numpy()
    input_dict2 = {"input": input2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D complex tensor, specify multiple dims
    input3 = (torch.randn(3, 4, 5) + 1j * torch.randn(3, 4, 5)).numpy()
    input_dict3 = {"input": input3, "dim": (0, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D float tensor, negative values, no dim specified
    input4 = torch.randn(2, 3, 4, 5).numpy()
    input_dict4 = {"input": input4, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor with a single element
    input5 = torch.randn(1).numpy()
    input_dict5 = {"input": input5, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 2D float tensor, specify dim=0
    input6 = torch.randn(5, 5).numpy()
    input_dict6 = {"input": input6, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 3D int tensor with dim as a tuple
    input7 = torch.randint(-10, 10, (2, 3, 4)).numpy()
    input_dict7 = {"input": input7, "dim": (0,1)}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.ifftshift"] = ifftshift_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.ifftshift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifftshift'.")

check_valid('torch.fft.ifftshift', generated_inputs['torch.fft.ifftshift'], lib="torch")
