
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_normal_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a simple tensor for std and an out tensor of correct shape and type
    mean = 0.5
    std = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.zeros(3).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: std with negative values.
    mean = 1.0
    std = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.zeros(3).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: std with a mix of positive and negative values.
    mean = 0.0
    std = torch.tensor([-1.0, 2.0, -3.0]).numpy()
    out = torch.zeros(3).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: std with zero values.
    mean = 2.0
    std = torch.tensor([0.0, 0.0, 0.0]).numpy()
    out = torch.zeros(3).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: std as a 2D tensor.
    mean = 0.5
    std = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.zeros((2,2)).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: std as a 3D tensor.
    mean = 1.5
    std = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    out = torch.zeros((2, 2, 2)).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different mean value.
    mean = -0.5
    std = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.zeros(3).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large std values.
    mean = 0.0
    std = torch.tensor([1000.0, 2000.0, 3000.0]).numpy()
    out = torch.zeros(3).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small std values.
    mean = 0.0
    std = torch.tensor([0.001, 0.002, 0.003]).numpy()
    out = torch.zeros(3).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: std with mixed data types (but all converted to numpy).
    mean = 0.5
    std = torch.tensor([1, 2.0, 3]).float().numpy() # Ensure std is float type in numpy
    out = torch.zeros(3).numpy()
    input_dict = {"mean": mean, "std": std, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.normal_2"] = torch_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.normal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_2'.")

check_valid('torch.normal', generated_inputs['torch.normal_2'], lib="torch", suffix=2)
