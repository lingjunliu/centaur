
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def log1p_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive values
    input1 = torch.tensor([0.1, 0.5, 1.0, 2.0]).numpy()
    out1 = torch.tensor([]).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor, negative values but > -1
    input2 = torch.tensor([-0.1, -0.5, -0.9]).numpy()
    out2 = torch.tensor([]).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, mixed positive and negative values
    input3 = torch.tensor([[0.2, -0.3], [-0.8, 1.5]]).numpy()
    out3 = torch.tensor([]).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, all positive
    input4 = torch.rand(2, 3, 4).numpy()
    out4 = torch.tensor([]).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor with a zero value
    input5 = torch.tensor([0.0, 0.5, -0.2]).numpy()
    out5 = torch.tensor([]).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = torch.tensor([]).numpy()
    out6 = torch.tensor([]).numpy()
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with large values
    input7 = torch.tensor([1000.0, 2000.0, 3000.0]).numpy()
    out7 = torch.tensor([]).numpy()
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log1p'.")

check_valid('torch.log1p', generated_inputs['torch.log1p'], lib="torch", suffix=0)
