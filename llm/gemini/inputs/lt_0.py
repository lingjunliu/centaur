
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lt_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other1 = torch.tensor([[1, 1], [4, 4]]).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Float tensor vs. a scalar
    input2 = torch.randn(3, 3).numpy()
    other2 = 0.5
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Broadcasting with different shapes
    input3 = torch.arange(5).numpy()
    other3 = torch.tensor([3]).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Negative values and different data types
    input4 = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).numpy()
    other4 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 3D tensors
    input5 = torch.randn(2, 3, 4).numpy()
    other5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Scalar vs Tensor
    input6 = 2.0
    other6 = torch.randn(2,2).numpy()
    input_dict6 = {"input": other6, "other": input6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: 1D Tensor
    input7 = torch.arange(10).numpy()
    other7 = torch.arange(10, 20).numpy()
    input_dict7 = {"input": input7, "other": other7, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.lt"] = lt_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lt'.")

check_valid('torch.lt', generated_inputs['torch.lt'], lib="torch")
