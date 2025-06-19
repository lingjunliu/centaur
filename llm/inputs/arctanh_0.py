
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def arctanh_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with values in the range (-1, 1)
    input_tensor = torch.tensor([-0.5, 0.0, 0.5]).numpy()
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with values in the range (-1, 1)
    input_tensor = torch.tensor([[-0.2, 0.3], [0.6, -0.8]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with values close to 1
    input_tensor = torch.tensor([[[0.9, 0.95], [0.8, 0.7]], [[0.5, 0.6], [0.4, 0.3]]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with a single element close to -1
    input_tensor = torch.tensor([-0.99]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with mixed positive and negative values.
    input_tensor = torch.tensor([-0.1, 0.2, -0.3, 0.4, -0.5]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: tensor with value zero
    input_tensor = torch.tensor([0.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: tensor with a large shape
    input_tensor = torch.rand(5, 5, 5).numpy() * 2 - 1
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.arctanh"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arctanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arctanh'.")

check_valid('torch.arctanh', generated_inputs['torch.arctanh'], lib="torch", suffix=0)
