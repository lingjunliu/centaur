
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_fft_ifftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, default dim
    input1 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    dim1 = (0,)
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, dim=0
    input2 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    dim2 = (0,)
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, dim=1
    input3 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    dim3 = (1,)
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D tensor, dim=(0, 1)
    input4 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    dim4 = (0, 1)
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor, dim=0
    input5 = torch.randn(2, 3, 4).numpy()
    dim5 = (0,)
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D tensor, dim=1
    input6 = torch.randn(2, 3, 4).numpy()
    dim6 = (1,)
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 3D tensor, dim=2
    input7 = torch.randn(2, 3, 4).numpy()
    dim7 = (2,)
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 3D tensor, dim=(0, 1)
    input8 = torch.randn(2, 3, 4).numpy()
    dim8 = (0, 1)
    input_dict8 = {"input": input8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 3D tensor, dim=(0, 2)
    input9 = torch.randn(2, 3, 4).numpy()
    dim9 = (0, 2)
    input_dict9 = {"input": input9, "dim": dim9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: 3D tensor, dim=(1, 2)
    input10 = torch.randn(2, 3, 4).numpy()
    dim10 = (1, 2)
    input_dict10 = {"input": input10, "dim": dim10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: 4D tensor, dim=(0, 1, 2, 3)
    input11 = torch.randn(2, 3, 4, 5).numpy()
    dim11 = (0, 1, 2, 3)
    input_dict11 = {"input": input11, "dim": dim11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: 1D tensor with negative values
    input12 = torch.tensor([-1, -2, -3, 4, 5]).numpy()
    dim12 = (0,)
    input_dict12 = {"input": input12, "dim": dim12}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.ifftshift_2"] = torch_fft_ifftshift_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.ifftshift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifftshift_2'.")

check_valid('torch.fft.ifftshift', generated_inputs['torch.fft.ifftshift_2'], lib="torch", suffix=2)
