
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def threshold_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor, positive threshold and value
    input_tensor = torch.tensor([1.0, 2.0, -1.0, 0.5, 3.0]).numpy()
    threshold_value = 1.5
    value_to_replace = 0.0
    input_dict = {"input": input_tensor, "threshold": threshold_value, "value": value_to_replace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, negative threshold and value
    input_tensor = torch.tensor([[-1.0, -2.0], [0.5, 1.5]]).numpy()
    threshold_value = -1.5
    value_to_replace = -2.0
    input_dict = {"input": input_tensor, "threshold": threshold_value, "value": value_to_replace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, zero threshold, different value
    input_tensor = torch.randn(2, 3, 4).numpy()
    threshold_value = 0.0
    value_to_replace = 1.0
    input_dict = {"input": input_tensor, "threshold": threshold_value, "value": value_to_replace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with only negative values
    input_tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    threshold_value = -2.5
    value_to_replace = 0.0
    input_dict = {"input": input_tensor, "threshold": threshold_value, "value": value_to_replace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with only positive values
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    threshold_value = 0.5
    value_to_replace = 10.0
    input_dict = {"input": input_tensor, "threshold": threshold_value, "value": value_to_replace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.threshold'.")

check_valid('torch.threshold', generated_inputs['torch.threshold'], lib="torch", suffix=0)
