
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def scatter_reduce_inputs():
    list_of_inputs = []

    # Input 1
    input = np.zeros((5,), dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    reduce = "sum"
    output_size = (5,)
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.ones((2, 3), dtype=np.float32)
    dim = 0
    index = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    src = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    reduce = "mean"
    output_size = (2, 3)
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.full((3, 2), 2.0, dtype=np.float32)
    dim = 1
    index = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int64)
    src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    reduce = "amin"
    output_size = (3, 2)
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.full((3, 2), 5.0, dtype=np.float32)
    dim = 1
    index = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int64)
    src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    reduce = "amax"
    output_size = (3, 2)
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.zeros((4,), dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 2, 0], dtype=np.int64)
    src = np.array([1.0, 2.0, 3.0, -4.0], dtype=np.float32)
    reduce = "sum"
    output_size = (4,)
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.full((2, 2), 2.0, dtype=np.float32)
    dim = 0
    index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    reduce = "prod"
    output_size = (2, 2)
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.ones((3,), dtype=np.float32)
    dim = 0
    index = np.array([0, 1, 0], dtype=np.int64)
    src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reduce = "sum"
    output_size = (3,)
    include_self = True

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.zeros((2, 3), dtype=np.float32)
    dim = 1
    index = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    src = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    reduce = "sum"
    output_size = (2, 3)
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = np.full((2, 2), 2.0, dtype=np.float32)
    dim = 0
    index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    reduce = "sum"
    output_size = (2, 2)
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.full((2, 2), 2.0, dtype=np.float32)
    dim = 1
    index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    reduce = "sum"
    output_size = (2, 2)
    include_self = False

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src,
        "reduce": reduce,
        "include_self": include_self,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.scatter_reduce_3"] = scatter_reduce_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.scatter_reduce_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter_reduce_3'.")

check_valid('torch.scatter_reduce', generated_inputs['torch.scatter_reduce_3'], lib="torch", suffix=3)
