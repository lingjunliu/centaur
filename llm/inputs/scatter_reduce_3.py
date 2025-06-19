
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def scatter_reduce_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    input_tensor = np.zeros((5,))
    dim = 0
    index_tensor = np.array([0, 1, 2, 0, 3])
    src_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    reduce = "sum"
    output_size = (5,)
    include_self = False

    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "src": src_tensor,
        "reduce": reduce,
        "output_size": output_size,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multi-dimensional input
    input_tensor = np.ones((3, 3))
    dim = 0
    index_tensor = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    src_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    reduce = "sum"
    output_size = (3, 3)
    include_self = True
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "src": src_tensor,
        "reduce": reduce,
        "output_size": output_size,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Use 'mean' reduction
    input_tensor = np.ones((5,))
    dim = 0
    index_tensor = np.array([0, 1, 2, 0, 3])
    src_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    reduce = "mean"
    output_size = (5,)
    include_self = True

    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "src": src_tensor,
        "reduce": reduce,
        "output_size": output_size,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different output size
    input_tensor = np.zeros((10,))
    dim = 0
    index_tensor = np.array([0, 1, 2, 0, 3, 4, 5, 6, 7, 8])
    src_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    reduce = "sum"
    output_size = (10,)
    include_self = False

    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "src": src_tensor,
        "reduce": reduce,
        "output_size": output_size,
        "include_self": include_self
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 'prod' reduction
    input_tensor = np.ones((5,))
    dim = 0
    index_tensor = np.array([0, 1, 2, 0, 3])
    src_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    reduce = "prod"
    output_size = (5,)
    include_self = True

    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "src": src_tensor,
        "reduce": reduce,
        "output_size": output_size,
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
