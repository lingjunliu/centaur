
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 4).astype(np.float32)
    target1 = np.random.randint(0, 2, size=(3, 4)).astype(np.float32)
    weight1 = np.random.rand(3, 4).astype(np.float32)
    pos_weight1 = np.random.rand(1).astype(np.float32)

    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "reduction": 'mean',
        "pos_weight": pos_weight1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(5).astype(np.float64)
    target2 = np.random.randint(0, 2, size=(5)).astype(np.float64)
    weight2 = np.random.rand(5).astype(np.float64)
    pos_weight2 = np.random.rand(1).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "reduction": 'sum',
        "pos_weight": pos_weight2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input4 = np.random.randn(1, 5, 5).astype(np.float32)
    target4 = np.random.randint(0, 2, size=(1, 5, 5)).astype(np.float32)
    weight4 = None
    pos_weight4 = None

    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": weight4,
        "reduction": 'mean',
        "pos_weight": pos_weight4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(2, 3).astype(np.float32)
    target5 = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
    weight5 = None
    pos_weight5 = np.random.rand(1).astype(np.float32)

    input_dict5 = {
        "input": input5,
        "target": target5,
        "weight": weight5,
        "reduction": 'sum',
        "pos_weight": pos_weight5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = binary_cross_entropy_with_logits_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('binaryCrossEntropyWithLogits', generated_inputs)
