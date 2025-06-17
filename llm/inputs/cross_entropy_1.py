
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 2D input and 1D target
    input1 = torch.randn(3, 5).numpy()
    target1 = np.array([1, 0, 4], dtype=np.int64)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.cross_entropy_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cross_entropy_1'.")

check_valid('torch.nn.functional.cross_entropy', generated_inputs['torch.nn.functional.cross_entropy_1'], lib="torch")
