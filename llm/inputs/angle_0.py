
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def angle_inputs():
    list_of_inputs = []

    # Input 1: 1D complex tensor
    input_tensor = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex tensor
    input_tensor = torch.tensor([[1 + 0j, 0 + 1j], [-1 + 0j, 0 - 1j]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D real tensor (should return 0 or pi based on sign)
    input_tensor = torch.tensor([-1.0, 0.0, 1.0, -0.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex tensor
    input_tensor = torch.randn(2, 2, 2, dtype=torch.complex64).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor with NaN
    input_tensor = torch.tensor([float('nan') + 1j, 1 + float('nan')*1j]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty out tensor
    input_tensor = torch.tensor([1 + 1j, -1 + 1j]).numpy()
    out_tensor = np.array([])
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero tensor
    input_tensor = torch.zeros(2, 2, dtype=torch.complex64).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.angle"] = angle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.angle'.")

check_valid('torch.angle', generated_inputs['torch.angle'], lib="torch", suffix=0)
