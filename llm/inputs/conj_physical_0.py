
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conj_physical_inputs():
    list_of_inputs = []

    # Input 1: 1D complex tensor
    input_tensor = torch.tensor([1 + 1j, 2 - 2j, 3 + 0j], dtype=torch.complex64).numpy()
    out_tensor = np.zeros(input_tensor.shape, dtype=np.complex64)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex tensor
    input_tensor = torch.tensor([[1 + 1j, 2 - 2j], [3 + 0j, 4 - 1j]], dtype=torch.complex128).numpy()
    out_tensor = np.zeros(input_tensor.shape, dtype=np.complex128)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D complex tensor
    input_tensor = torch.tensor([[[1 + 1j, 2 - 2j], [3 + 0j, 4 - 1j]], [[5 - 1j, 6 + 2j], [7 + 0j, 8 + 1j]]], dtype=torch.complex64).numpy()
    out_tensor = np.zeros(input_tensor.shape, dtype=np.complex64)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex tensor with only real parts
    input_tensor = torch.tensor([1 + 0j, 2 + 0j, 3 + 0j], dtype=torch.complex128).numpy()
    out_tensor = np.zeros(input_tensor.shape, dtype=np.complex128)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor with only imaginary parts
    input_tensor = torch.tensor([0 + 1j, 0 - 2j, 0 + 3j], dtype=torch.complex64).numpy()
    out_tensor = np.zeros(input_tensor.shape, dtype=np.complex64)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large complex tensor
    input_tensor = torch.randn(10, 10, dtype=torch.complex128).numpy()
    out_tensor = np.zeros(input_tensor.shape, dtype=np.complex128)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.conj_physical"] = conj_physical_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.conj_physical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.conj_physical'.")

check_valid('torch.conj_physical', generated_inputs['torch.conj_physical'], lib="torch", suffix=0)
