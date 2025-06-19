
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def psi_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive values
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    out = torch.tensor([]).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with positive values
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.tensor([]).numpy()
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor with zero
    input = torch.tensor([0.0]).numpy()
    out = torch.tensor([]).numpy()
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: multi-dimensional tensor
    input = torch.randn(2, 3, 4).numpy()
    out = torch.tensor([]).numpy()
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor with a range of values
    input = torch.arange(1.0, 10.0, step=0.5).numpy()
    out = torch.tensor([]).numpy()
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.psi"] = psi_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.psi' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.psi'.")

check_valid('torch.special.psi', generated_inputs['torch.special.psi'], lib="torch", suffix=0)
