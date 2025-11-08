
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def binary_cross_entropy_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([0.1, 0.9, 0.4], dtype=torch.float32).numpy()
    target = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32).numpy()
    weight = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.tensor([[0.2, 0.8, 0.6],
                              [0.05, 0.95, 0.5]], dtype=torch.float32).numpy()
    target = torch.tensor([[0.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0]], dtype=torch.float32).numpy()
    weight = np.ones((2, 3), dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": False,
        "reduce": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.tensor([[[0.3, 0.7],
                               [0.4, 0.6]]], dtype=torch.float64).numpy()
    target = torch.tensor([[[0.0, 1.0],
                            [1.0, 0.0]]], dtype=torch.float64).numpy()
    weight = torch.tensor([[[1.0, 0.5],
                            [2.0, 1.5]]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": True,
        "reduce": False,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    base = np.array([[0.10, 0.20, 0.30, 0.40],
                     [0.50, 0.60, 0.70, 0.80],
                     [0.90, 0.05, 0.15, 0.25]], dtype=np.float32)
    inp4 = np.stack([base, base * 0.9], axis=0)
    inp4 = np.expand_dims(inp4, axis=1)
    tgt4 = np.stack([(base > 0.5).astype(np.float32),
                     (base * 0.9 > 0.5).astype(np.float32)], axis=0)
    tgt4 = np.expand_dims(tgt4, axis=1)
    w4 = np.full_like(inp4, 1.2, dtype=np.float32)
    input_dict = {
        "input": inp4,
        "target": tgt4,
        "weight": w4,
        "size_average": False,
        "reduce": False,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    weight = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = np.array([[0.2, 0.9],
                          [0.8, 0.1]], dtype=np.float32)
    target = np.array([[0.0, 1.0],
                       [1.0, 0.0]], dtype=np.float32)
    weight = np.array([[-1.0, -0.5],
                       [-2.0, -1.5]], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": True,
        "reduce": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.array([1e-6, 1 - 1e-6, 1e-4, 0.999, 0.01], dtype=np.float64)
    target = np.array([0.0, 1.0, 0.5, 1.0, 0.0], dtype=np.float64)
    weight = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": False,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (broadcastable weight across last dim)
    input_arr = np.array([[0.1, 0.2, 0.3, 0.4],
                          [0.6, 0.7, 0.8, 0.9],
                          [0.25, 0.5, 0.75, 0.35]], dtype=np.float32)
    target = np.array([[0.0, 0.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    weight = np.array([[1.0],
                       [0.5],
                       [2.0]], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.linspace(0.05, 0.95, num=2*3*4*5, dtype=np.float32).reshape(2, 3, 4, 5)
    target = ((np.arange(2*3*4*5) % 2) == 0).astype(np.float32).reshape(2, 3, 4, 5)
    weight = np.ones((2, 3, 4, 5), dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": False,
        "reduce": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = np.array([[0.7]], dtype=np.float32)
    target = np.array([[1.0]], dtype=np.float32)
    weight = np.array([[2.5]], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": True,
        "reduce": False,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = np.array([[[0.1, 0.9]],
                          [[0.3, 0.7]],
                          [[0.6, 0.4]],
                          [[0.8, 0.2]]], dtype=np.float64)
    target = np.array([[[0.0, 1.0]],
                       [[0.0, 1.0]],
                       [[1.0, 0.0]],
                       [[1.0, 0.0]]], dtype=np.float64)
    weight = np.array([[[1.0, 0.5]],
                       [[2.0, 1.5]],
                       [[0.7, 0.3]],
                       [[1.2, 0.8]]], dtype=np.float64)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = np.array([0.12, 0.34, 0.56, 0.78, 0.91, 0.23], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    weight = np.array([1.0, 0.5, 2.0, 1.5, 0.1, 3.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target,
        "weight": weight,
        "size_average": False,
        "reduce": False,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.binary_cross_entropy"] = binary_cross_entropy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.binary_cross_entropy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.binary_cross_entropy'.")


check_valid('torch.nn.functional.binary_cross_entropy', generated_inputs['torch.nn.functional.binary_cross_entropy'], lib="torch", suffix=0)
