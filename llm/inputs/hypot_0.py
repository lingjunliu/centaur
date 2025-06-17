
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hypot_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = torch.tensor([4.0]).numpy()
    input2 = torch.tensor([3.0, 4.0, 5.0]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Broadcasting with scalars
    input1 = torch.tensor(5.0).numpy()
    input2 = torch.tensor([[3.0, 4.0], [5.0, 12.0]]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Multi-dimensional tensors
    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(2, 3).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Negative values
    input1 = torch.tensor([-3.0, 4.0]).numpy()
    input2 = torch.tensor([4.0, -3.0]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Different shapes that are broadcastable
    input1 = torch.randn(3, 1).numpy()
    input2 = torch.randn(1, 3).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.hypot"] = hypot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hypot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hypot'.")

check_valid('torch.hypot', generated_inputs['torch.hypot'], lib="torch")
