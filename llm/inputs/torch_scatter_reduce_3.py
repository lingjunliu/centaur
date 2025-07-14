
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def scatter_reduce_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D example with sum reduction
    input = np.zeros(5, dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 0, 2, 1], dtype=np.int64)
    src = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    reduce = "sum"
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D example with mean reduction and include_self=True
    input = np.ones((2, 3), dtype=np.float32)
    dim = 1
    index = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    src = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32)
    reduce = "mean"
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D example with max reduction
    input = np.zeros((2, 2, 2), dtype=np.float32)
    dim = 0
    index = np.array([[[0, 1], [0, 1]], [[1, 0], [1, 0]]], dtype=np.int64)
    src = np.array([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]], dtype=np.float32)
    reduce = "max"
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D example with min reduction and negative values
    input = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 0], dtype=np.int64)
    src = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    reduce = "min"
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D example with mul reduction
    input = np.ones((2, 3), dtype=np.float32)
    dim = 1
    index = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    src = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32)
    reduce = "mul"
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Example with output_size larger than input size in dimension
    input = np.zeros(3, dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 0], dtype=np.int64)
    src = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    reduce = "sum"
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Another 2D example with different index
    input = np.zeros((2, 3), dtype=np.float32)
    dim = 1
    index = np.array([[1, 2, 1], [0, 1, 0]], dtype=np.int64)
    src = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32)
    reduce = "sum"
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D example with amin reduction
    input = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 0], dtype=np.int64)
    src = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    reduce = "amin"
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D example with amax reduction
    input = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 0], dtype=np.int64)
    src = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    reduce = "amax"
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D example with amax reduction, with include_self = False
    input = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 0], dtype=np.int64)
    src = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    reduce = "amax"
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.scatter_reduce_3"] = scatter_reduce_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.scatter_reduce_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter_reduce_3'.")

check_valid('torch.scatter_reduce', generated_inputs['torch.scatter_reduce_3'], lib="torch", suffix=3)
