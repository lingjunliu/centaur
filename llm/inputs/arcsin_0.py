
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def arcsin_inputs():
    list_of_inputs = []

    # Input 1, valid: basic 1D tensor
    input = torch.tensor([0.5]).numpy()
    out = torch.tensor([0.0]).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid: 1D tensor with negative values
    input = torch.tensor([-0.2, 0.0, 0.3, -0.7]).numpy()
    out = torch.tensor([-0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid: 2D tensor
    input = torch.tensor([[0.1, 0.2], [-0.3, -0.4]]).numpy()
    out = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid: 3D tensor
    input = torch.tensor([[[0.5, -0.5], [0.2, -0.2]], [[0.1, -0.1], [0.8, -0.8]]]).numpy()
    out = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid: Tensor with values close to 1 and -1
    input = torch.tensor([0.99, -0.99, 0.01, -0.01]).numpy()
    out = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid: Empty tensor
    input = torch.tensor([]).numpy()
    out = torch.tensor([]).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid: specific shape
    input = torch.randn(2, 2, 3).numpy() * 0.5
    out = torch.zeros(2, 2, 3).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid: different data type
    input = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64).numpy()
    out = torch.zeros(3, dtype=torch.float64).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.arcsin"] = arcsin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.arcsin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arcsin'.")

check_valid('torch.arcsin', generated_inputs['torch.arcsin'], lib="torch", suffix=0)
