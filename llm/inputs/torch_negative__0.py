
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def negative_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, float32, out specified
    input_tensor = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, int64, out specified
    input_tensor = torch.tensor([[1, 2], [-3, 4]], dtype=torch.int64)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, float64, out specified
    input_tensor = torch.randn(2, 3, 4, dtype=torch.float64)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor, int32, out specified
    input_tensor = torch.tensor([-1, 0, 1], dtype=torch.int32)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor, float16, out specified
    input_tensor = torch.tensor([[0.5, -1.0], [2.0, -0.25]], dtype=torch.float16)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor, float32, out specified
    input_tensor = torch.tensor([], dtype=torch.float32)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar tensor, int8, out specified
    input_tensor = torch.tensor(-5, dtype=torch.int8)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor, complex64, out specified
    input_tensor = torch.tensor([1+1j, 2-2j, -3+0j], dtype=torch.complex64)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D tensor, complex128, out specified
    input_tensor = torch.tensor([[1+1j, 2-2j], [-3+0j, 0+4j]], dtype=torch.complex128)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large tensor, float32, out specified
    input_tensor = torch.randn(100, 100, dtype=torch.float32)
    out_tensor = torch.empty_like(input_tensor)
    input = input_tensor.numpy()
    out = out_tensor.numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.negative_"] = negative_inputs()

def check_valid(api, inputs, lib="torch", suffix=0):
    for i, input_dict in enumerate(inputs):
        try:
            input_tensor = torch.from_numpy(input_dict["input"])
            out_tensor = torch.from_numpy(input_dict["out"])
            torch.negative_(input=input_tensor, out=out_tensor)
        except Exception as e:
            print(f"Error on input {i+1}: {e}")
            raise e

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.negative_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.negative_'.")

check_valid('torch.negative_', generated_inputs['torch.negative_'], lib="torch", suffix=0)
