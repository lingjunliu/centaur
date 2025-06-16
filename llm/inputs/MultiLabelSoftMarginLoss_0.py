
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MultiLabelSoftMarginLoss_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default parameters
    input1 = torch.randn(2, 3).numpy()
    target1 = torch.randint(0, 2, (2, 3)).numpy()
    input_dict1 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: With weight
    input2 = torch.randn(2, 3).numpy()
    target2 = torch.randint(0, 2, (2, 3)).numpy()
    weight2 = torch.randn(3).numpy()
    input_dict2 = {
        "weight": weight2,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: With different reduction
    input3 = torch.randn(2, 3).numpy()
    target3 = torch.randint(0, 2, (2, 3)).numpy()
    input_dict3 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: With 'none' reduction
    input4 = torch.randn(2, 3).numpy()
    target4 = torch.randint(0, 2, (2, 3)).numpy()
    input_dict4 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger batch size
    input5 = torch.randn(5, 4).numpy()
    target5 = torch.randint(0, 2, (5, 4)).numpy()
    input_dict5 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different input range
    input6 = torch.randn(2, 3) * 100.0
    target6 = torch.randint(0, 2, (2, 3)).numpy()
    input_dict6 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input6.numpy(),
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MultiLabelSoftMarginLoss"] = MultiLabelSoftMarginLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MultiLabelSoftMarginLoss', generated_inputs['torch.nn.MultiLabelSoftMarginLoss'], lib="torch")
