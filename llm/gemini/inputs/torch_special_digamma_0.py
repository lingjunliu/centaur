
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def digamma_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive values
    input_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out_1 = np.array([], dtype=np.float32)
    input_dict_1 = {"input": input_1, "out": out_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor, positive values
    input_2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out_2 = np.array([], dtype=np.float64)
    input_dict_2 = {"input": input_2, "out": out_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D tensor, mixed values (positive and small)
    input_3 = np.array([0.1, 0.5, 1.0], dtype=np.float32)
    out_3 = np.array([], dtype=np.float32)
    input_dict_3 = {"input": input_3, "out": out_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensor, positive values
    input_4 = np.random.rand(2, 3, 4).astype(np.float64)
    out_4 = np.array([], dtype=np.float64)
    input_dict_4 = {"input": input_4, "out": out_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1D tensor, larger positive values
    input_5 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    out_5 = np.array([], dtype=np.float32)
    input_dict_5 = {"input": input_5, "out": out_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D tensor with different data type
    input_6 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    out_6 = np.array([], dtype=np.float64)
    input_dict_6 = {"input": input_6, "out": out_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 1D tensor with a single element
    input_7 = np.array([5.0], dtype=np.float32)
    out_7 = np.array([], dtype=np.float32)
    input_dict_7 = {"input": input_7, "out": out_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2D tensor with different shapes
    input_8 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    out_8 = np.array([], dtype=np.float64)
    input_dict_8 = {"input": input_8, "out": out_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 1D tensor with large values
    input_9 = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    out_9 = np.array([], dtype=np.float32)
    input_dict_9 = {"input": input_9, "out": out_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D tensor
    input_10 = np.random.rand(2, 2, 2, 2).astype(np.float64)
    out_10 = np.array([], dtype=np.float64)
    input_dict_10 = {"input": input_10, "out": out_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.special.digamma"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.digamma'.")

check_valid('torch.special.digamma', generated_inputs['torch.special.digamma'], lib="torch", suffix=0)
