
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def softmarginloss_inputs():
    list_of_inputs = []

    input1 = np.array([0.5, -1.2, 2.0, -0.3], dtype=np.float32)
    target1 = np.array([1, -1, 1, -1], dtype=np.float32)
    input_dict1 = {
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.8, -0.5], [-1.5, 1.2]], dtype=np.float32)
    target2 = np.array([[1, -1], [-1, 1]], dtype=np.float32)
    input_dict2 = {
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.2, -0.9, 1.5], dtype=np.float32)
    target3 = np.array([1, -1, 1], dtype=np.float32)
    input_dict3 = {
        "size_average": None,
        "reduce": False,
        "reduction": 'none',
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(2, 3, 4).astype(np.float32)
    target4 = np.random.choice([-1, 1], size=(2, 3, 4)).astype(np.float32)
    input_dict4 = {
        "size_average": True,
        "reduce": False,
        "reduction": 'none',
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(1, 5).astype(np.float32)
    target5 = np.random.choice([-1, 1], size=(1, 5)).astype(np.float32)
    input_dict5 = {
        "size_average": False,
        "reduce": None,
        "reduction": 'mean',
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-2.0, 1.0], [0.5, -0.8]], dtype=np.float32)
    target6 = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    input_dict6 = {
        "size_average": None,
        "reduce": True,
        "reduction": 'mean',
        "input": input6,
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([1.5, -0.7, 0.0], dtype=np.float32)
    target7 = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    input_dict7 = {
        "size_average": None,
        "reduce": True,
        "reduction": 'sum',
        "input": input7,
        "target": target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.SoftMarginLoss"] = softmarginloss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.SoftMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SoftMarginLoss'.")

check_valid('torch.nn.SoftMarginLoss', generated_inputs['torch.nn.SoftMarginLoss'], lib="torch")
