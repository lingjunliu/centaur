
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rsqrt_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive values
    input1 = torch.tensor([1.0, 4.0, 9.0, 16.0]).numpy()
    out1 = torch.tensor([]).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with positive values
    input2 = torch.tensor([[1.0, 4.0], [9.0, 16.0]]).numpy()
    out2 = torch.tensor([]).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with positive values
    input3 = torch.tensor([[[1.0, 4.0], [9.0, 16.0]], [[25.0, 36.0], [49.0, 64.0]]]).numpy()
    out3 = torch.tensor([]).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with a mix of positive and zero values
    input4 = torch.tensor([0.0, 1.0, 4.0, 0.0]).numpy()
    out4 = torch.tensor([]).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Tensor with only zero values
    input5 = torch.zeros((2, 2)).numpy()
    out5 = torch.tensor([]).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.rsqrt"] = rsqrt_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rsqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsqrt'.")

check_valid('torch.rsqrt', generated_inputs['torch.rsqrt'], lib="torch", suffix=0)
