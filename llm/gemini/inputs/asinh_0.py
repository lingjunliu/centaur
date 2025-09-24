
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def asinh_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive and negative values
    input_tensor = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with different values
    input_tensor = torch.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with small values
    input_tensor = torch.randn(2, 2, 2) * 0.1
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with zero values
    input_tensor = torch.zeros(3, 3)
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large values
    input_tensor = torch.tensor([-10.0, -5.0, 5.0, 10.0])
    out_tensor = torch.empty_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 tensor
    input_tensor = torch.randn(2, 2, dtype=torch.float64)
    out_tensor = np.empty_like(input_tensor.numpy())
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty Tensor
    input_tensor = torch.tensor([])
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.asinh"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.asinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asinh'.")

check_valid('torch.asinh', generated_inputs['torch.asinh'], lib="torch", suffix=0)
