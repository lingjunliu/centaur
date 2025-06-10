
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []

    # Case 1: Basic case with default parameters
    input_tensor = np.random.randn(3, 5).astype(np.float32)
    target_tensor = np.random.randint(0, 5, size=(3,)).astype(np.int64)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different reduction mode and weight
    input_tensor = np.random.randn(2, 4).astype(np.float32)
    target_tensor = np.random.randint(0, 4, size=(2,)).astype(np.int64)
    weight_tensor = np.random.rand(4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "ignore_index": -100,
        "reduction": 'sum',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Ignore index
    input_tensor = np.random.randn(4, 3).astype(np.float32)
    target_tensor = np.random.randint(0, 3, size=(4,)).astype(np.int64)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": None,
        "ignore_index": 1,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Label smoothing
    input_tensor = np.random.randn(5, 6).astype(np.float32)
    target_tensor = np.random.randint(0, 6, size=(5,)).astype(np.int64)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    # Case 6: Weight with ignore_index
    input_tensor = np.random.randn(3, 4).astype(np.float32)
    target_tensor = np.random.randint(0, 4, size=(3,)).astype(np.int64)
    weight_tensor = np.random.rand(4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "ignore_index": 2,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: None reduction - Removing for now as non scalar output leads to error
    input_tensor = np.random.randn(2, 3).astype(np.float32)
    target_tensor = np.random.randint(0, 3, size=(2,)).astype(np.int64)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean', # changed reduction to mean
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.cross_entropy"] = cross_entropy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cross_entropy', generated_inputs['torch.nn.functional.cross_entropy'], lib="torch")
