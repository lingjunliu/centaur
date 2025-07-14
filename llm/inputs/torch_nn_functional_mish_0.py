
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def mish_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor, inplace=False
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict1 = {"input": input1, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Simple 1D tensor, inplace=True
    input2 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict2 = {"input": input2, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, inplace=False
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict3 = {"input": input3, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D tensor with negative values, inplace=True
    input4 = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    input_dict4 = {"input": input4, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor, inplace=False
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {"input": input5, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D tensor with a mix of positive and negative values, inplace=True
    input6 = torch.randn(2, 3, 4).numpy()
    input_dict6 = {"input": input6, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with zeros, inplace=False
    input7 = torch.zeros(5).numpy()
    input_dict7 = {"input": input7, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Tensor with ones, inplace=True
    input8 = torch.ones(5).numpy()
    input_dict8 = {"input": input8, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Larger tensor, inplace = False
    input9 = torch.randn(10, 10).numpy()
    input_dict9 = {"input": input9, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Tensor with large values, inplace=True
    input10 = torch.tensor([100.0, -100.0, 50.0, -50.0]).numpy()
    input_dict10 = {"input": input10, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.mish"] = mish_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.mish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.mish'.")

check_valid('torch.nn.functional.mish', generated_inputs['torch.nn.functional.mish'], lib="torch", suffix=0)
