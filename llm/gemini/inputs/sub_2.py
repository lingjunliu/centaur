
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sub_inputs():
    list_of_inputs = []

    # Case 1: Basic case with int tensors
    input = torch.tensor([1, 2, 3]).numpy()
    other = 1.0
    alpha = 1.0
    out = torch.empty(3).numpy()
    input_dict = {"input": input, "other": other, "alpha": alpha, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors with negative values and different alpha
    input = torch.tensor([1.5, -2.5, 3.5]).numpy()
    other = 0.5
    alpha = 2.0
    out = torch.empty(3).numpy()
    input_dict = {"input": input, "other": other, "alpha": alpha, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Broadcasting with scalar 'other'
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    other = 2.0
    alpha = 1.0
    out = torch.empty((2,2)).numpy()
    input_dict = {"input": input, "other": other, "alpha": alpha, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Broadcasting with 1D 'other'
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    other = np.array([0.5, 1.0])
    alpha = 1.0
    out = torch.empty((2,2)).numpy()
    input_dict = {"input": input, "other": other, "alpha": alpha, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D tensor
    input = torch.randn(2, 3, 4).numpy()
    other = 0.1
    alpha = 1.0
    out = torch.empty((2, 3, 4)).numpy()
    input_dict = {"input": input, "other": other, "alpha": alpha, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Complex Tensor
    input = torch.complex(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, -2.0, 0.0])).numpy()
    other = 0.5
    alpha = 1.0
    out = torch.empty(3, dtype=torch.complex64).numpy()
    input_dict = {"input": input, "other": other, "alpha": alpha, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Zero values
    input = torch.tensor([0, 0, 0]).numpy()
    other = 1.0
    alpha = 1.0
    out = torch.empty(3).numpy()
    input_dict = {"input": input, "other": other, "alpha": alpha, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sub_2"] = sub_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sub_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sub_2'.")

check_valid('torch.sub', generated_inputs['torch.sub_2'], lib="torch")
