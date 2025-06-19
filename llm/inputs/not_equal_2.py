
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def not_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor and float value
    input_tensor = torch.randn(3, 4).numpy()
    other_value = 0.5
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor and float value
    input_tensor = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    other_value = -1.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Bool tensor and float value
    input_tensor = torch.tensor([[True, False], [False, True]]).numpy()
    other_value = 0.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float tensor and float value
    input_tensor = torch.randn(5).numpy()
    other_value = 0.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D float tensor and float value
    input_tensor = torch.randn(2, 3, 4).numpy()
    other_value = 1.2
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative float tensor and float value
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy()
    other_value = -0.5
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero tensor and float value
    input_tensor = torch.zeros(3, 3).numpy()
    other_value = 0.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.not_equal_2"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.not_equal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.not_equal_2'.")

check_valid('torch.not_equal', generated_inputs['torch.not_equal_2'], lib="torch")
