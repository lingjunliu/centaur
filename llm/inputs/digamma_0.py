
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def digamma_inputs():
    list_of_inputs = []

    # Input 1: Positive values, 1D tensor
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive values, 2D tensor
    input_tensor = torch.tensor([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive values, 3D tensor
    input_tensor = torch.rand(2, 3, 4).numpy() + 0.5 # Ensure positive values
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger positive values, 1D
    input_tensor = torch.tensor([10.0, 20.0, 30.0, 40.0]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small positive values, 2D
    input_tensor = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed positive values
    input_tensor = torch.tensor([0.5, 1.0, 2.5, 5.0]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar tensor
    input_tensor = torch.tensor(3.14).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.digamma"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.digamma'.")

check_valid('torch.digamma', generated_inputs['torch.digamma'], lib="torch", suffix=0)
