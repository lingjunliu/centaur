
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([[0.5, -1.2, 3.0, 0.0],
                          [-0.7, 2.3, -0.1, 1.5],
                          [1.2, -0.5, 0.8, -2.0]], dtype=np.float32)
    target_arr = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 0, 0]], dtype=np.float32)
    weight_arr = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.array([[2.0, -3.0, 0.5, 1.2, -0.8]], dtype=np.float32)
    target_arr = np.array([[1, 0, 1, 0, 1]], dtype=np.float32)
    weight_arr = np.array([0.2, 1.0, 1.5, 0.7, 2.5], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": False,
        "reduce": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.array([0.1, -0.2, 3.4, -4.5, 0.0, 1.1], dtype=np.float32)
    target_arr = np.array([0, 1, 1, 0, 0, 1], dtype=np.float32)
    weight_arr = np.array([1.0, 1.0, 0.5, 2.0, 1.0, 0.3], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": True,
        "reduce": False,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = np.array([[1.5, -1.5, 0.0],
                          [-0.3, 0.8, -2.2]], dtype=np.float64)
    target_arr = np.array([[1, 0, 0],
                           [0, 1, 0]], dtype=np.float64)
    weight_arr = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": False,
        "reduce": False,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = np.array([[0.0, 2.2],
                          [-1.0, -0.5],
                          [3.0, 0.5],
                          [0.7, -2.1],
                          [-0.9, 1.8]], dtype=np.float32)
    target_arr = np.array([[0, 1],
                           [1, 0],
                           [1, 1],
                           [0, 0],
                           [0, 1]], dtype=np.float32)
    weight_arr = np.array([0.0, 1.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = np.array([-5.0, 4.0, -3.0], dtype=np.float32)
    target_arr = np.array([0, 1, 1], dtype=np.float32)
    weight_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": False,
        "reduce": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.array([[0.2, -0.1, 0.3, -0.4, 1.0, -1.0, 2.0],
                          [1.5, 2.2, -2.3, 0.0, -0.7, 0.9, -1.8],
                          [-0.6, 0.4, 1.1, -1.2, 0.8, -0.2, 0.5],
                          [2.0, -2.0, 0.0, 1.0, -1.0, 1.5, -0.5]], dtype=np.float32)
    target_arr = np.array([[1, 0, 1, 0, 1, 0, 1],
                           [0, 1, 0, 1, 0, 1, 0],
                           [1, 1, 0, 0, 1, 0, 0],
                           [0, 0, 1, 1, 0, 1, 1]], dtype=np.float32)
    weight_arr = np.array([1.0, 0.8, 1.2, 0.5, 1.5, 1.0, 0.7], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = np.array([[3.0, -3.0, 0.1, -0.1],
                          [-2.0, 2.0, -0.5, 0.5]], dtype=np.float32)
    target_arr = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1]], dtype=np.float32)
    weight_arr = np.array([2.0, 2.0, 0.5, 0.5], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": True,
        "reduce": False,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.array([[0.5],
                          [-1.5],
                          [2.0],
                          [0.0],
                          [1.2],
                          [-0.7],
                          [3.3],
                          [-2.1],
                          [0.9],
                          [-0.4]], dtype=np.float32)
    target_arr = np.array([[1],
                           [0],
                           [1],
                           [0],
                           [1],
                           [0],
                           [1],
                           [0],
                           [1],
                           [0]], dtype=np.float32)
    weight_arr = np.array([1.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = np.array([[-0.25]], dtype=np.float32)
    target_arr = np.array([[1]], dtype=np.float32)
    weight_arr = np.array([3.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": False,
        "reduce": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = np.array([[1.0, -2.0, 0.5, 1.5, -0.5],
                          [-1.0, 2.0, -0.5, -1.5, 0.5]], dtype=np.float64)
    target_arr = np.array([[1, 0, 1, 1, 0],
                           [0, 1, 0, 0, 1]], dtype=np.float64)
    weight_arr = np.array([0.9, 1.1, 0.7, 1.3, 0.6], dtype=np.float64)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": False,
        "reduce": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = np.array([[0.0, 0.0, 0.0],
                          [1.0, -1.0, 2.0],
                          [-2.0, 2.0, -2.0]], dtype=np.float32)
    target_arr = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype=np.float32)
    weight_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "target": target_arr,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.multilabel_soft_margin_loss"] = multilabel_soft_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.multilabel_soft_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.multilabel_soft_margin_loss'.")


check_valid('torch.nn.functional.multilabel_soft_margin_loss', generated_inputs['torch.nn.functional.multilabel_soft_margin_loss'], lib="torch", suffix=0)
