
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def NLLLoss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 'mean' reduction
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 'sum' reduction
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'sum',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 'none' reduction
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'none',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: with weight
    input_dict = {
        "weight": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy(),
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: with ignore_index
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": 0,
        "reduce": None,
        "reduction": 'mean',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.NLLLoss"] = NLLLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.NLLLoss', generated_inputs['torch.nn.NLLLoss'], lib="torch")
