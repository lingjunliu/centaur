
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def one_hot_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input = np.array([0, 1, 2], dtype=np.int64)
    num_classes = 3
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input = np.array([[0, 1], [2, 0]], dtype=np.int64)
    num_classes = 3
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input = np.array([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=np.int64)
    num_classes = 3
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different num_classes
    input = np.array([0, 1, 2], dtype=np.int64)
    num_classes = 5
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Values close to num_classes
    input = np.array([0, 1, 2], dtype=np.int64)
    num_classes = 4
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element tensor
    input = np.array(0, dtype=np.int64)
    num_classes = 3
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with all zeros
    input = np.array([0, 0, 0, 0], dtype=np.int64)
    num_classes = 5
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional tensor with values smaller than num_classes
    input = np.array([[0, 2], [1, 0]], dtype=np.int64)
    num_classes = 5
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor, different values
    input = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    num_classes = 4
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Example with different shapes
    input = np.array([1, 0, 3, 2], dtype=np.int64)
    num_classes = 5
    input_dict = {"input": input, "num_classes": num_classes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.one_hot"] = one_hot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.one_hot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.one_hot'.")

check_valid('torch.nn.functional.one_hot', generated_inputs['torch.nn.functional.one_hot'], lib="torch", suffix=0)
