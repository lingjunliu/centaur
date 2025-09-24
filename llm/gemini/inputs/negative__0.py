
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def negative__inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive values
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor, negative values
    input_tensor = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor, mixed values
    input_tensor = torch.tensor([-1.0, 2.0, -3.0, 4.0])
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor, mixed values
    input_tensor = torch.tensor([[[1.0, -2.0], [-3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]])
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float 32 tensor
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float 64 tensor
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.negative_"] = negative__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.negative_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.negative_'.")

check_valid('torch.negative_', generated_inputs['torch.negative_'], lib="torch", suffix=0)
