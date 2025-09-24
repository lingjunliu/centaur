
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def exp2_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor, positive values, out is None
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out1 = torch.zeros_like(torch.tensor([1.0, 2.0, 3.0])).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    # Input 2: 2D tensor, mixed positive and negative values, out is provided
    input2 = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).numpy()
    out2 = torch.zeros_like(torch.tensor([[-1.0, 0.0], [1.0, 2.0]])).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3: 3D tensor, large values
    input3 = torch.tensor([[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]]).numpy()
    out3 = torch.zeros_like(torch.tensor([[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]])).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor, negative values
    input4 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out4 = torch.zeros_like(torch.tensor([-1.0, -2.0, -3.0])).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 0D tensor, scalar value
    input5 = torch.tensor(5.0).numpy()
    out5 = torch.zeros_like(torch.tensor(5.0)).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.exp2"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.exp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.exp2'.")

check_valid('torch.exp2', generated_inputs['torch.exp2'], lib="torch", suffix=0)
