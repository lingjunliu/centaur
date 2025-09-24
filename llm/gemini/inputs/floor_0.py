
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def floor_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive and negative floats
    input_tensor = torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with different float values
    input_tensor = torch.tensor([[1.2, 2.7], [-0.3, -1.8]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with mixed values
    input_tensor = torch.tensor([[[1.1, -2.2], [3.3, -4.4]], [[5.5, -6.6], [7.7, -8.8]]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with only zero
    input_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large tensor
    input_tensor = torch.randn(100).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.floor"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor'.")

check_valid('torch.floor', generated_inputs['torch.floor'], lib="torch", suffix=0)
