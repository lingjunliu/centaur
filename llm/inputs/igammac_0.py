
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def igammac_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other_tensor = torch.tensor([0.5, 1.0, 1.5]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different values and out tensor size
    input_tensor = torch.tensor([0.5, 1.5, 2.5]).numpy()
    other_tensor = torch.tensor([2.0, 1.0, 0.5]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional tensors
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other_tensor = torch.tensor([[0.5, 1.0], [1.5, 2.0]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger values
    input_tensor = torch.tensor([5.0, 10.0, 15.0]).numpy()
    other_tensor = torch.tensor([2.5, 5.0, 7.5]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different data types
    input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()
    other_tensor = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: More dimensions
    input_tensor = torch.randn(2, 3, 4).numpy()
    other_tensor = torch.randn(2, 3, 4).numpy()
    out_tensor = torch.zeros(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All ones.
    input_tensor = np.ones((2, 2))
    other_tensor = np.ones((2, 2))
    out_tensor = np.zeros((2, 2))
    input_dict = {"input": input_tensor, "other": other_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.igammac"] = igammac_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.igammac' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.igammac'.")

check_valid('torch.igammac', generated_inputs['torch.igammac'], lib="torch", suffix=0)
