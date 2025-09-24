
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def BCEWithLogitsLoss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with no optional parameters
    input1 = np.random.randn(3, 4).astype(np.float32)
    target1 = np.random.randint(0, 2, size=(3, 4)).astype(np.float32)
    input_dict1 = {"input": input1, "target": target1, "weight": None, "size_average": None, "reduce": None, "reduction": 'mean', "pos_weight": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: With pos_weight
    input2 = np.random.randn(2, 5).astype(np.float32)
    target2 = np.random.randint(0, 2, size=(2, 5)).astype(np.float32)
    pos_weight2 = np.random.rand(5).astype(np.float32)
    input_dict2 = {"input": input2, "target": target2, "weight": None, "size_average": None, "reduce": None, "reduction": 'mean', "pos_weight": pos_weight2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: With reduction='sum'
    input3 = np.random.randn(4, 2).astype(np.float32)
    target3 = np.random.randint(0, 2, size=(4, 2)).astype(np.float32)
    input_dict3 = {"input": input3, "target": target3, "weight": None, "size_average": None, "reduce": None, "reduction": 'sum', "pos_weight": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: With reduction='none'
    input4 = np.random.randn(1, 3).astype(np.float32)
    target4 = np.random.randint(0, 2, size=(1, 3)).astype(np.float32)
    input_dict4 = {"input": input4, "target": target4, "weight": None, "size_average": None, "reduce": None, "reduction": 'none', "pos_weight": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: With weight
    input5 = np.random.randn(2, 3, 4).astype(np.float32)
    target5 = np.random.randint(0, 2, size=(2, 3, 4)).astype(np.float32)
    weight5 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict5 = {"input": input5, "target": target5, "weight": weight5, "size_average": None, "reduce": None, "reduction": 'mean', "pos_weight": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Higher dimension input and target
    input6 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    target6 = np.random.randint(0, 2, size=(2, 3, 4, 5)).astype(np.float32)
    input_dict6 = {"input": input6, "target": target6, "weight": None, "size_average": None, "reduce": None, "reduction": 'mean', "pos_weight": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.BCEWithLogitsLoss"] = BCEWithLogitsLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.BCEWithLogitsLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BCEWithLogitsLoss'.")

check_valid('torch.nn.BCEWithLogitsLoss', generated_inputs['torch.nn.BCEWithLogitsLoss'], lib="torch")
