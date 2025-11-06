
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def multimarginloss_inputs():
    list_of_inputs = []

    # 1
    input_arr = np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32)
    target_arr = np.array([3], dtype=np.int64)
    weight_arr = np.ones((4,), dtype=np.float32)
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = np.array([[0.5, -0.3, 1.2, 0.0, -0.7],
                          [1.5, 0.2, -0.4, 0.9, 0.1],
                          [-1.0, 0.5, 0.3, -0.2, 1.0]], dtype=np.float64)
    target_arr = np.array([2, 0, 4], dtype=np.int64)
    weight_arr = np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float64)
    input_dict = {
        "p": 2,
        "margin": 1.0,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = np.array([0.3, -0.1, 0.5], dtype=np.float32)
    target_arr = np.array(2, dtype=np.int64)
    weight_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = np.array([[-1.0, 0.0, 2.0],
                          [0.5, -0.5, 1.5]], dtype=np.float64)
    target_arr = np.array([2, 0], dtype=np.int64)
    weight_arr = np.ones((3,), dtype=np.float64)
    input_dict = {
        "p": 1,
        "margin": 0.5,
        "weight": weight_arr,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = np.array([[0.2, -1.0, 0.5, 1.2, -0.3, 0.0],
                          [1.1, 0.7, -0.8, 0.4, 0.9, -0.2],
                          [-0.5, 0.2, 1.5, -1.2, 0.3, 0.8],
                          [0.0, -0.4, 0.6, 0.1, -0.9, 1.3]], dtype=np.float32)
    target_arr = np.array([0, 3, 2, 5], dtype=np.int64)
    weight_arr = np.array([1.0, 0.5, 1.5, 2.0, 0.1, 3.0], dtype=np.float32)
    input_dict = {
        "p": 2,
        "margin": 2.0,
        "weight": weight_arr,
        "size_average": True,
        "reduce": False,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = np.array([[-2.0, -1.0]], dtype=np.float32)
    target_arr = np.array([1], dtype=np.int64)
    weight_arr = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {
        "p": 1,
        "margin": 1.5,
        "weight": weight_arr,
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = np.array([[0.1, -0.2, 0.3, 0.4],
                          [1.0, 0.5, -0.5, 0.0],
                          [-0.7, 1.2, 0.0, -0.1],
                          [0.8, -1.1, 0.6, 0.2],
                          [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    target_arr = np.array([3, 0, 1, 2, 0], dtype=np.int64)
    weight_arr = np.array([0.1, 1.0, 5.0, 2.0], dtype=np.float32)
    input_dict = {
        "p": 2,
        "margin": 0.1,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = np.array([-0.2, 0.0, 1.2, 0.5, -1.5], dtype=np.float64)
    target_arr = np.array(2, dtype=np.int64)
    weight_arr = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float64)
    input_dict = {
        "p": 2,
        "margin": 1.0,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = np.zeros((2, 2), dtype=np.float32)
    target_arr = np.array([0, 1], dtype=np.int64)
    weight_arr = np.array([1.0, 2.0], dtype=np.float32)
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = np.array([[0.9, -0.2, 0.1, 0.0, 0.3, -0.5, 0.7, 1.1],
                          [-0.1, 0.4, -0.6, 0.8, -0.3, 0.2, 0.0, -0.9],
                          [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
    target_arr = np.array([7, 0, 3], dtype=np.int64)
    weight_arr = np.ones((8,), dtype=np.float32)
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = np.array([[0.1, -0.2, 0.3],
                          [0.0, 0.5, -0.1],
                          [-0.3, 0.2, 0.8],
                          [1.0, -1.0, 0.0],
                          [0.4, 0.4, 0.4],
                          [-0.5, 0.9, -0.2],
                          [0.7, 0.1, -0.8],
                          [0.2, -0.4, 1.0],
                          [-1.2, 0.3, 0.9],
                          [0.6, -0.7, 0.0]], dtype=np.float64)
    target_arr = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 1], dtype=np.int64)
    weight_arr = np.array([0.5, 0.7, 1.2], dtype=np.float64)
    input_dict = {
        "p": 2,
        "margin": 1.0,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = np.array([0.0, 0.0], dtype=np.float32)
    target_arr = np.array(0, dtype=np.int64)
    weight_arr = np.array([2.5, 0.5], dtype=np.float32)
    input_dict = {
        "p": 1,
        "margin": 0.2,
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MultiMarginLoss"] = multimarginloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MultiMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MultiMarginLoss'.")


check_valid('torch.nn.MultiMarginLoss', generated_inputs['torch.nn.MultiMarginLoss'], lib="torch", suffix=0)
