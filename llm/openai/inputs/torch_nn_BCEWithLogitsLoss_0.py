
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bcewithlogitsloss_inputs():
    list_of_inputs = []

    # Input 1: 1D
    input_arr = torch.tensor([0.5, -1.2, 2.3], dtype=torch.float32).numpy()
    target_arr = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32).numpy()
    weight = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32).numpy()
    pos_weight = torch.tensor([2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": True, "reduction": "mean",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 2: 2D
    input_arr = torch.randn(2, 4, dtype=torch.float32).numpy()
    target_arr = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0, 1.0]], dtype=torch.float32).numpy()
    weight = torch.ones(2, 4, dtype=torch.float32).numpy()
    pos_weight = torch.tensor([1.0, 2.0, 3.0, 0.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": False, "reduce": True, "reduction": "sum",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 3: 2D (single batch)
    input_arr = torch.tensor([[0.2, -0.7, 1.5, -2.0, 0.0]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.5]], dtype=torch.float32).numpy()
    weight = torch.full((1, 5), 0.7, dtype=torch.float32).numpy()
    pos_weight = torch.ones(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": False, "reduction": "mean",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 4: 4D
    input_arr = torch.randn(2, 3, 4, 4, dtype=torch.float32).numpy()
    target_arr = torch.randint(0, 2, (2, 3, 4, 4), dtype=torch.float32).numpy()
    weight = torch.linspace(0.5, 1.5, steps=2*3*4*4, dtype=torch.float32).reshape(2, 3, 4, 4).numpy()
    pos_weight = torch.tensor([3.0, 1.0, 0.5], dtype=torch.float32).view(3, 1, 1).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": True, "reduction": "mean",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 5: 1D with extremes
    input_arr = torch.tensor([10.0, -10.0, 20.0, -20.0, 0.0], dtype=torch.float32).numpy()
    target_arr = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32).numpy()
    weight = torch.ones(5, dtype=torch.float32).numpy()
    pos_weight = torch.tensor([5.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": True, "reduction": "mean",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 6: 2D with C=1
    input_arr = torch.tensor([[0.0], [-2.0]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[0.9], [0.1]], dtype=torch.float32).numpy()
    weight = torch.tensor([[1.5], [0.5]], dtype=torch.float32).numpy()
    pos_weight = torch.tensor([0.7], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": True, "reduction": "none",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 7: 4D with spatial broadcasting of pos_weight
    input_arr = torch.randn(3, 2, 2, 3, dtype=torch.float32).numpy()
    target_arr = torch.rand(3, 2, 2, 3, dtype=torch.float32).numpy()
    weight = torch.rand(3, 2, 2, 3, dtype=torch.float32).numpy()
    pos_weight = torch.tensor([1.2, 0.8], dtype=torch.float32).view(2, 1, 1).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": True, "reduction": "mean",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 8: 1D with fractional targets
    input_arr = torch.tensor([-3.0, 0.3, 1.2], dtype=torch.float32).numpy()
    target_arr = torch.tensor([0.2, 0.7, 0.9], dtype=torch.float32).numpy()
    weight = torch.tensor([0.3, 2.0, 1.5], dtype=torch.float32).numpy()
    pos_weight = torch.tensor([2.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": False, "reduce": False, "reduction": "none",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 9: 2D multi-label
    input_arr = torch.tensor([[0.3, -0.8, 1.2],
                              [-1.5, 0.7, -0.2]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[0.2, 0.7, 0.9],
                               [0.8, 0.1, 0.4]], dtype=torch.float32).numpy()
    weight = torch.tensor([[1.0, 0.5, 1.5],
                           [0.8, 1.2, 0.7]], dtype=torch.float32).numpy()
    pos_weight = torch.tensor([2.0, 0.5, 1.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": False, "reduce": False, "reduction": "none",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 10: 4D NCHW with C=2
    input_arr = torch.randn(1, 2, 1, 1, dtype=torch.float32).numpy()
    target_arr = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float32).numpy()
    weight = torch.ones(1, 2, 1, 1, dtype=torch.float32).numpy()
    pos_weight = torch.tensor([4.0, 0.5], dtype=torch.float32).view(2, 1, 1).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": True, "reduction": "sum",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 11: 2D with zeros pos_weight (no positive emphasis)
    input_arr = torch.randn(2, 4, dtype=torch.float32).numpy()
    target_arr = torch.randint(0, 2, (2, 4), dtype=torch.float32).numpy()
    weight = torch.full((2, 4), 1.0, dtype=torch.float32).numpy()
    pos_weight = torch.zeros(4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": True, "reduction": "mean",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    # Input 12: 3D with pos_weight broadcast across last dim
    input_arr = torch.randn(2, 3, 2, dtype=torch.float32).numpy()
    target_arr = torch.randint(0, 2, (2, 3, 2), dtype=torch.float32).numpy()
    weight = torch.linspace(0.8, 1.2, steps=2*3*2, dtype=torch.float32).reshape(2, 3, 2).numpy()
    pos_weight = torch.tensor([[1.0], [2.0], [0.5]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "weight": weight, "size_average": True, "reduce": True, "reduction": "mean",
        "pos_weight": pos_weight, "input": input_arr, "target": target_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.BCEWithLogitsLoss"] = bcewithlogitsloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BCEWithLogitsLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BCEWithLogitsLoss'.")


check_valid('torch.nn.BCEWithLogitsLoss', generated_inputs['torch.nn.BCEWithLogitsLoss'], lib="torch", suffix=0)
