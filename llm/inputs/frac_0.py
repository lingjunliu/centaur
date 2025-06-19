
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def frac_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive and negative floats
    input_tensor = torch.tensor([1.5, -2.3, 0.7, -0.1]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with integers
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with mixed values
    input_tensor = torch.tensor([[[1.1, 2.2], [-3.3, 4.4]], [[-5.5, 6.6], [7.7, -8.8]]]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D tensor with zeros and near-zero values
    input_tensor = torch.tensor([0.0, 1e-9, -1e-9, 1.0]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with large values
    input_tensor = torch.tensor([1e9 + 0.5, -1e9 - 0.5]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.frac"] = frac_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.frac' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.frac'.")

check_valid('torch.frac', generated_inputs['torch.frac'], lib="torch", suffix=0)
