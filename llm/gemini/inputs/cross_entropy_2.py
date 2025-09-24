
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 2D input and 1D target
    input = torch.randn(3, 5).numpy()
    target = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input, "target": target, "weight": None, "ignore_index": -100, "reduction": 'mean', "label_smoothing": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: With weight
    input = torch.randn(3, 5).numpy()
    target = torch.randint(0, 5, (3,)).numpy()
    weight = torch.rand(5).numpy()
    input_dict = {"input": input, "target": target, "weight": weight, "ignore_index": -100, "reduction": 'mean', "label_smoothing": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: With ignore_index
    input = torch.randn(3, 5).numpy()
    target = np.array([0, 1, -100])
    input_dict = {"input": input, "target": target, "weight": None, "ignore_index": -100, "reduction": 'mean', "label_smoothing": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: With different reduction
    input = torch.randn(3, 5).numpy()
    target = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input, "target": target, "weight": None, "ignore_index": -100, "reduction": 'sum', "label_smoothing": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: With label smoothing
    input = torch.randn(3, 5).numpy()
    target = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input, "target": target, "weight": None, "ignore_index": -100, "reduction": 'mean', "label_smoothing": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.cross_entropy_2"] = cross_entropy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.cross_entropy_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cross_entropy_2'.")

check_valid('torch.nn.functional.cross_entropy', generated_inputs['torch.nn.functional.cross_entropy_2'], lib="torch")
