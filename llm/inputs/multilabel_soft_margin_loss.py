
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic example with float tensors
    input_tensor = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target_tensor = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    weight_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Example with different reduction method
    input_tensor = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    target_tensor = np.array([[1, 0], [0, 1]], dtype=np.float32)
    weight_tensor = None
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Example with no reduction
    input_tensor = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target_tensor = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    weight_tensor = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": False,
        "reduce": False,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Example with negative values
    input_tensor = np.array([[-0.1, 0.2], [0.4, -0.5]], dtype=np.float32)
    target_tensor = np.array([[0, 1], [1, 0]], dtype=np.float32)
    weight_tensor = None
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Example with scalar weight
    input_tensor = np.array([[0.6, 0.7], [0.8, 0.9]], dtype=np.float32)
    target_tensor = np.array([[1, 0], [0, 1]], dtype=np.float32)
    weight_tensor = np.array([2.0, 0.5], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 3D input and target
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    target_tensor = np.random.randint(0, 2, size=(2, 3, 4)).astype(np.float32)
    weight_tensor = np.random.rand(4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Example with different data
    input_tensor = np.array([[0.7, -0.8], [-0.9, 1.0]], dtype=np.float32)
    target_tensor = np.array([[0, 1], [1, 0]], dtype=np.float32)
    weight_tensor = None
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = multilabel_soft_margin_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('multilabel_soft_margin_loss', generated_inputs)
