
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float input and long target
    input = np.random.randn(3, 5).astype(np.float32)
    target = np.random.randint(0, 5, size=(3,)).astype(np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2:  weight is specified
    input = np.random.randn(3, 5).astype(np.float32)
    target = np.random.randint(0, 5, size=(3,)).astype(np.int64)
    weight = np.random.rand(5).astype(np.float32)
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: ignore_index is specified
    input = np.random.randn(3, 5).astype(np.float32)
    target = np.random.randint(-1, 5, size=(3,)).astype(np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -1,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: reduction = 'sum'
    input = np.random.randn(3, 5).astype(np.float32)
    target = np.random.randint(0, 5, size=(3,)).astype(np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'sum',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: reduction = 'none'
    input = np.random.randn(3, 5).astype(np.float32)
    target = np.random.randint(0, 5, size=(3,)).astype(np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'none',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: label_smoothing > 0
    input = np.random.randn(3, 5).astype(np.float32)
    target = np.random.randint(0, 5, size=(3,)).astype(np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Different input shape
    input = np.random.randn(1, 10).astype(np.float32)
    target = np.random.randint(0, 10, size=(1,)).astype(np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.cross_entropy_3"] = cross_entropy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.cross_entropy_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cross_entropy_3'.")

check_valid('torch.nn.functional.cross_entropy', generated_inputs['torch.nn.functional.cross_entropy_3'], lib="torch")
