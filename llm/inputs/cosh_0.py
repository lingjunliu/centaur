
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cosh_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor, out=None
    input_tensor = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, specified out tensor
    input_tensor = torch.tensor([[0.0, 1.0], [-1.0, 2.0]]).numpy()
    out_tensor = torch.empty_like(torch.tensor(input_tensor)).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with large values (positive and negative)
    input_tensor = torch.tensor([-100.0, 100.0, -500.0, 500.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with a mix of small and large values
    input_tensor = torch.tensor([-0.1, 0.1, -10.0, 10.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with zeros
    input_tensor = torch.zeros((2, 2)).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Empty Tensor
    input_tensor = torch.tensor([]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cosh"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cosh'.")

check_valid('torch.cosh', generated_inputs['torch.cosh'], lib="torch", suffix=0)
