
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def MultiLabelSoftMarginLoss_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 5).astype(np.float32)
    target1 = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    weight1 = np.random.rand(5).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 4).astype(np.float64)
    target2 = np.random.randint(0, 2, size=(2, 4)).astype(np.float64)
    weight2 = None
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(1, 7).astype(np.float32)
    target3 = np.random.randint(0, 2, size=(1, 7)).astype(np.float32)
    weight3 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "target": target3,
        "weight": weight3,
        "size_average": False,
        "reduce": False,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(4, 3, 2).astype(np.float32)
    target4 = np.random.randint(0, 2, size=(4, 3, 2)).astype(np.float32)
    weight4 = None
    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": weight4,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    target5 = np.random.randint(0, 2, size=(2, 2, 2, 2)).astype(np.float32)
    weight5 = None
    input_dict5 = {
        "input": input5,
        "target": target5,
        "weight": weight5,
        "size_average": False,
        "reduce": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = MultiLabelSoftMarginLoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MultiLabelSoftMarginLoss', generated_inputs)
