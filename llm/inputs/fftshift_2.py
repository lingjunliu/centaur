
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_fft_fftshift_inputs():
    list_of_inputs = []

    # Example 1: 1D float tensor
    input1 = torch.fft.fftfreq(4).numpy()
    input_dict1 = {"input": input1, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 2D float tensor, shifting only one dimension
    input2 = torch.fft.fftfreq(5, d=1/5) + 0.1 * torch.fft.fftfreq(5, d=1/5).unsqueeze(1)
    input2 = input2.numpy()
    input_dict2 = {"input": input2, "dim": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 3D complex tensor, shifting multiple dimensions
    input3 = (torch.randn(3, 4, 5) + 1j * torch.randn(3, 4, 5)).numpy()
    input_dict3 = {"input": input3, "dim": (0, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: 2D integer tensor
    input4 = torch.arange(-5, 5).reshape(2, 5).numpy()
    input_dict4 = {"input": input4, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: 1D float tensor with negative values
    input5 = torch.linspace(-1, 1, 7).numpy()
    input_dict5 = {"input": input5, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.fft.fftshift_2"] = torch_fft_fftshift_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.fftshift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fftshift_2'.")

check_valid('torch.fft.fftshift', generated_inputs['torch.fft.fftshift_2'], lib="torch")
