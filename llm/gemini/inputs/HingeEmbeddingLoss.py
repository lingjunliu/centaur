
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def hinge_embedding_loss_inputs():
    list_of_inputs = []

    input1 = np.array([0.5, 0.8, 0.2, 0.9]).astype(np.float32)
    target1 = np.array([1, -1, 1, -1]).astype(np.int8)
    input_dict1 = {
        "margin": 1.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append({"input": input1, "target": target1, **input_dict1})

    input2 = np.array([-0.5, -0.8, -0.2, -0.9]).astype(np.float32)
    target2 = np.array([1, -1, 1, -1]).astype(np.int8)
    input_dict2 = {
        "margin": 0.5,
        "size_average": True,
        "reduce": False,
        "reduction": 'sum'
    }
    list_of_inputs.append({"input": input2, "target": target2, **input_dict2})

    input3 = np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float32)
    target3 = np.array([[1, -1], [-1, 1]]).astype(np.int8)
    input_dict3 = {
        "margin": 2.0,
        "size_average": False,
        "reduce": True,
        "reduction": 'none'
    }
    list_of_inputs.append({"input": input3, "target": target3, **input_dict3})

    input4 = np.array([0.7]).astype(np.float64)
    target4 = np.array([-1]).astype(np.int8)
    input_dict4 = {
        "margin": 1.5,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append({"input": input4, "target": target4, **input_dict4})

    input5 = np.array([0.2, 0.4, 0.6]).astype(np.float32)
    target5 = np.array([1, 1, 1]).astype(np.int8)
    input_dict5 = {
        "margin": 1.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append({"input": input5, "target": target5, **input_dict5})

    input6 = np.array([0.5, 0.8, 0.2, 0.9]).astype(np.float32)
    target6 = np.array([1, -1, 1, -1]).astype(np.int8)
    input_dict6 = {
        "margin": 1.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'none'
    }
    list_of_inputs.append({"input": input6, "target": target6, **input_dict6})
    
    input7 = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    target7 = np.array([1, 1, -1]).astype(np.int8)
    input_dict7 = {
        "margin": 0.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append({"input": input7, "target": target7, **input_dict7})

    return list_of_inputs

generated_inputs = hinge_embedding_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('HingeEmbeddingLoss', generated_inputs)
