
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, dim=0, keepdim=False
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    dim1 = [0]
    keepdim1 = False
    out1 = torch.tensor([]).numpy()

    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, dim=0, keepdim=True
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim2 = [0]
    keepdim2 = True
    out2 = torch.tensor([]).numpy()

    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, dim=1, keepdim=False
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim3 = [1]
    keepdim3 = False
    out3 = torch.tensor([]).numpy()

    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, dim=[0, 1], keepdim=True
    input4 = torch.randn(2, 3, 4).numpy()
    dim4 = [0, 1]
    keepdim4 = True
    out4 = torch.tensor([]).numpy()

    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor, dim=[1, 2], keepdim=False
    input5 = torch.randn(2, 3, 4).numpy()
    dim5 = [1, 2]
    keepdim5 = False
    out5 = torch.tensor([]).numpy()

    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1D tensor with negative values, dim=0, keepdim=False
    input6 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    dim6 = [0]
    keepdim6 = False
    out6 = torch.tensor([]).numpy()

    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "keepdim": keepdim6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 2D tensor with negative values, dim=0, keepdim=True
    input7 = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    dim7 = [0]
    keepdim7 = True
    out7 = torch.tensor([]).numpy()

    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "keepdim": keepdim7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 4D tensor, dim=[0, 2], keepdim=False
    input8 = torch.randn(2, 3, 4, 5).numpy()
    dim8 = [0, 2]
    keepdim8 = False
    out8 = torch.tensor([]).numpy()

    input_dict8 = {
        "input": input8,
        "dim": dim8,
        "keepdim": keepdim8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 4D tensor, dim=[1, 3], keepdim=True
    input9 = torch.randn(2, 3, 4, 5).numpy()
    dim9 = [1, 3]
    keepdim9 = True
    out9 = torch.tensor([]).numpy()

    input_dict9 = {
        "input": input9,
        "dim": dim9,
        "keepdim": keepdim9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: Empty tensor
    input10 = torch.tensor([]).numpy()
    dim10 = []
    keepdim10 = False
    out10 = torch.tensor([]).numpy()

    input_dict10 = {
        "input": input10,
        "dim": dim10,
        "keepdim": keepdim10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.special.logsumexp_2"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.logsumexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.logsumexp_2'.")

check_valid('torch.special.logsumexp', generated_inputs['torch.special.logsumexp_2'], lib="torch", suffix=2)
