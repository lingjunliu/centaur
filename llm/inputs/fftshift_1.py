
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input_1 = torch.arange(-5, 5, dtype=torch.float32).numpy()
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D complex tensor
    real_part = torch.randn(3, 4)
    imag_part = torch.randn(3, 4)
    input_2 = torch.complex(real_part, imag_part).numpy()
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D float tensor with even dimensions
    input_3 = torch.randn(4, 6).numpy()
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.fftshift_1"] = fftshift_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.fftshift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fftshift_1'.")

check_valid('torch.fft.fftshift', generated_inputs['torch.fft.fftshift_1'], lib="torch")
