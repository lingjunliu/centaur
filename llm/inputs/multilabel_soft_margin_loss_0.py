
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensors, default reduction
    input1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target1 = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: With weight tensor, sum reduction
    input2 = np.array([[-0.1, 0.2], [0.4, -0.5]], dtype=np.float64)
    target2 = np.array([[1, 0], [0, 1]], dtype=np.float64)
    weight2 = np.array([0.5, 0.5], dtype=np.float64)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Different input size, no reduction
    input3 = np.array([[0.7, -0.8, 0.9, -0.1], [-0.2, 0.3, -0.4, 0.5]], dtype=np.float32)
    target3 = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)
    input_dict3 = {
        "input": input3,
        "target": target3,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 3D Input
    input4 = np.random.rand(2, 3, 4).astype(np.float32)
    target4 = np.random.randint(0, 2, size=(2, 3, 4)).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Another weight with different dimension.
    input5 = np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float32)
    target5 = np.array([[0, 1], [1, 0]], dtype=np.float32)
    weight5 = np.array([0.2, 0.8], dtype=np.float32)
    input_dict5 = {
        "input": input5,
        "target": target5,
        "weight": weight5,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs["torch.nn.functional.multilabel_soft_margin_loss"] = multilabel_soft_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.multilabel_soft_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.multilabel_soft_margin_loss'.")

check_valid('torch.nn.functional.multilabel_soft_margin_loss', generated_inputs['torch.nn.functional.multilabel_soft_margin_loss'], lib="torch")
