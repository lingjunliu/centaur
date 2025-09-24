
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def multilabelmarginloss_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32)
    target_tensor = np.array([[3, 0, -1, 1]], dtype=np.int64)
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[0.5, 0.3, 0.9, 0.1]], dtype=np.float32)
    target_tensor = np.array([[0, 2, -1, -1]], dtype=np.int64)
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": 'sum',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[0.7, 0.2, 0.5]], dtype=np.float32)
    target_tensor = np.array([[1, 0, -1]], dtype=np.int64)
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": 'none',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                             [0.6, 0.7, 0.8, 0.9, 1.0]], dtype=np.float32)
    target_tensor = np.array([[4, 2, 0, -1, -1],
                              [2, 3, 1, -1, -1]], dtype=np.int64)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[0.9, 0.8, 0.7, 0.6, 0.5],
                             [0.4, 0.3, 0.2, 0.1, 0.0]], dtype=np.float32)
    target_tensor = np.array([[0, 1, 2, -1, -1],
                              [4, 3, 2, -1, -1]], dtype=np.int64)
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[0.1, 0.3, 0.5]], dtype=np.float32)
    target_tensor = np.array([[2, 0, -1]], dtype=np.int64)
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": 'none',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)
    target_tensor = np.array([[0, 3, -1, -1]], dtype=np.int64)
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[0.8, 0.6, 0.4, 0.2]], dtype=np.float32)
    target_tensor = np.array([[3, 1, -1, -1]], dtype=np.int64)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": 'sum',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    target_tensor = np.array([[0, 1, 2]], dtype=np.int64)
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": 'none',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[0.2, 0.4, 0.6, 0.8, 1.0],
                             [0.1, 0.3, 0.5, 0.7, 0.9]], dtype=np.float32)
    target_tensor = np.array([[0, 2, 4, -1, -1],
                              [1, 3, -1, -1, -1]], dtype=np.int64)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    target_tensor = np.array([[3, 2, 1, 0]], dtype=np.int64)
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": "sum",
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MultiLabelMarginLoss"] = multilabelmarginloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MultiLabelMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MultiLabelMarginLoss'.")

check_valid('torch.nn.MultiLabelMarginLoss', generated_inputs['torch.nn.MultiLabelMarginLoss'], lib="torch", suffix=0)
