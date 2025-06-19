
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fix_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive values
    input1 = torch.tensor([1.2, 2.7, 3.1]).numpy()
    out1 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor, negative values
    input2 = torch.tensor([-1.2, -2.7, -3.1]).numpy()
    out2 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, mixed values
    input3 = torch.tensor([[-1.2, 2.7], [-3.1, 4.5]]).numpy()
    out3 = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, positive values
    input4 = torch.tensor([[[1.2, 2.7], [3.1, 4.5]], [[5.6, 6.7], [7.8, 9.0]]]).numpy()
    out4 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor, zero values
    input5 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    out5 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fix"] = fix_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fix'.")

check_valid('torch.fix', generated_inputs['torch.fix'], lib="torch", suffix=0)
