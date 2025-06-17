
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def gt_inputs():
    list_of_inputs = []

    input_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    other_value = 2.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([[-1, -2], [-3, -4]]).numpy()
    other_value = -2.5
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3, 4).numpy()
    other_value = 0.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randint(0, 10, (5,)).numpy()
    other_value = 5.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    other_value = 2.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.zeros(2, 2).numpy()
    other_value = 0.0
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.ones(3, 3).numpy()
    other_value = 0.5
    input_dict = {"input": input_tensor, "other": other_value, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.gt_2"] = gt_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.gt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gt_2'.")

check_valid('torch.gt', generated_inputs['torch.gt_2'], lib="torch")
