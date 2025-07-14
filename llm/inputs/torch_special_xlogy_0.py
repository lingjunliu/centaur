
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def xlogy_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: x = 0
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: y = 0
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: x = 0 and y = 0
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: y = 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional arrays
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    out = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different data type (float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    out = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Broadcasting.
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    out = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different shapes
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    out = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger values
    x = np.array([1e5, 2e5, 3e5], dtype=np.float32)
    y = np.array([4e5, 5e5, 6e5], dtype=np.float32)
    out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.xlogy"] = xlogy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.xlogy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlogy'.")

check_valid('torch.special.xlogy', generated_inputs['torch.special.xlogy'], lib="torch", suffix=0)
