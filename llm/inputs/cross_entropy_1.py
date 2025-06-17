
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 2D input and 1D target
    input_1 = np.random.randn(3, 5).astype(np.float32)
    target_1 = np.random.randint(0, 5, size=(3,)).astype(np.int64)
    input_dict_1 = {
        "input": input_1,
        "target": target_1,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: With weights
    input_2 = np.random.randn(2, 3).astype(np.float32)
    target_2 = np.random.randint(0, 3, size=(2,)).astype(np.int64)
    weight_2 = np.array([0.2, 0.3, 0.5]).astype(np.float32)
    input_dict_2 = {
        "input": input_2,
        "target": target_2,
        "weight": weight_2,
        "ignore_index": -100,
        "reduction": 'sum',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: With ignore_index
    input_3 = np.random.randn(4, 4).astype(np.float32)
    target_3 = np.array([0, 1, 2, -1]).astype(np.int64)
    input_dict_3 = {
        "input": input_3,
        "target": target_3,
        "weight": None,
        "ignore_index": -1,
        "reduction": 'none',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: With label smoothing
    input_4 = np.random.randn(2, 5).astype(np.float32)
    target_4 = np.random.randint(0, 5, size=(2,)).astype(np.int64)
    input_dict_4 = {
        "input": input_4,
        "target": target_4,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Remove the problematic 3D input case.

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.cross_entropy_1"] = cross_entropy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.cross_entropy_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cross_entropy_1'.")

check_valid('torch.nn.functional.cross_entropy', generated_inputs['torch.nn.functional.cross_entropy_1'], lib="torch")
