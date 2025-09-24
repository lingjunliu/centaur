
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sub_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([4, 5, 6]).numpy()
    alpha_val = 1
    out_tensor = None
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors with broadcasting
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other_tensor = torch.tensor([0.5, 1.0]).numpy()
    alpha_val = 2.0
    out_tensor = None
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values and different alpha
    input_tensor = torch.tensor([-1, -2, -3]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    alpha_val = -1
    out_tensor = None
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Scalar other (Number)
    input_tensor = torch.tensor([5, 6, 7]).numpy()
    other_tensor = 2
    alpha_val = 1
    out_tensor = None
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Higher dimensional tensors
    input_tensor = torch.randn(2, 3, 4).numpy()
    other_tensor = torch.randn(2, 3, 4).numpy()
    alpha_val = 1.5
    out_tensor = None
    input_dict = {"input": input_tensor, "other": other_tensor, "alpha": alpha_val, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sub_1"] = sub_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sub_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sub_1'.")

check_valid('torch.sub', generated_inputs['torch.sub_1'], lib="torch")
