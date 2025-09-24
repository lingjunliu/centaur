
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def trunc_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor
    input_tensor = torch.tensor([1.5, 2.7, -3.2, -0.8]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input_tensor = torch.tensor([[1.2, -2.3], [3.4, -4.5]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = torch.randn(2, 2, 2).numpy()
    out_tensor = torch.zeros_like(torch.randn(2, 2, 2)).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with only positive values
    input_tensor = torch.tensor([0.1, 1.2, 2.3, 3.4]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with only negative values
    input_tensor = torch.tensor([-0.1, -1.2, -2.3, -3.4]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with zero
    input_tensor = torch.tensor([0.0, 1.5, -2.5]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large values
    input_tensor = torch.tensor([100.5, -200.7]).numpy()
    out_tensor = torch.tensor([0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.trunc"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.trunc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.trunc'.")

check_valid('torch.trunc', generated_inputs['torch.trunc'], lib="torch", suffix=0)
