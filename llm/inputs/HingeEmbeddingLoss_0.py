
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def HingeEmbeddingLoss_inputs():
    list_of_inputs = []

    input1 = np.array([0.5, 0.8, 0.2, 0.9], dtype=np.float32)
    target1 = np.array([1, -1, 1, -1], dtype=np.int32)
    input_dict1 = {
        "margin": 1.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.3, 0.7], [0.1, 0.5]], dtype=np.float32)
    target2 = np.array([[1, -1], [-1, 1]], dtype=np.int32)
    input_dict2 = {
        "margin": 0.5,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.9, 0.4, 0.6], dtype=np.float32)
    target3 = np.array([1, 1, 1], dtype=np.int32)
    input_dict3 = {
        "margin": 1.5,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([0.2, 0.6, 0.8, 0.3], dtype=np.float32)
    target4 = np.array([-1, -1, -1, -1], dtype=np.int32)
    input_dict4 = {
        "margin": 1.0,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean',
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0.7], dtype=np.float32)
    target5 = np.array([-1], dtype=np.int32)
    input_dict5 = {
        "margin": 0.7,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.HingeEmbeddingLoss"] = HingeEmbeddingLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.HingeEmbeddingLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.HingeEmbeddingLoss'.")

check_valid('torch.nn.HingeEmbeddingLoss', generated_inputs['torch.nn.HingeEmbeddingLoss'], lib="torch")
