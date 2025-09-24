
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def fmin_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors
    input1 = torch.randint(0, 10, (2, 2)).numpy()
    input2 = torch.randint(5, 15, (2, 2)).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Broadcasting with different shapes
    input1 = torch.randn(5,).numpy()
    input2 = torch.randn(1, 5).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Including NaN values
    input1 = torch.tensor([1.0, float('nan'), 3.0, 4.0]).numpy()
    input2 = torch.tensor([5.0, 2.0, float('nan'), 8.0]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values and different data types
    input1 = torch.randn(2, 3).double().numpy()
    input2 = torch.randint(-10, 0, (2, 3)).float().numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 1D tensors
    input1 = torch.randn(10).numpy()
    input2 = torch.randn(10).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Scalar tensors
    input1 = torch.tensor(5.0).numpy()
    input2 = torch.tensor(2.0).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: All NaNs
    input1 = torch.tensor([float('nan'), float('nan')]).numpy()
    input2 = torch.tensor([float('nan'), float('nan')]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fmin"] = fmin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fmin'.")

check_valid('torch.fmin', generated_inputs['torch.fmin'], lib="torch")
