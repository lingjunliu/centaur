
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def max_unpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1.0]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0]]]]], dtype=np.int64)
    output_size = (1, 1, 1, 1, 1)
    stride = 1
    padding = 0
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0, 1], [2, 3]]]]], dtype=np.int64)
    output_size = (1, 1, 1, 2, 2)
    stride = 1
    padding = 0
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0, 3], [2, 5]]]]], dtype=np.int64)
    output_size = (1, 1, 1, 4, 4)
    stride = 2
    padding = 0
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_tensor = np.array([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0, 1], [2, 3]]]]], dtype=np.int64)
    output_size = (1, 1, 1, 3, 3)
    stride = 1
    padding = 1
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]]], dtype=np.int64)
    output_size = (1, 1, 2, 2, 2)
    stride = 1
    padding = 0
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[[1.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0]]]]], dtype=np.int64)
    output_size = (1, 1, 1, 5, 5)
    stride = 4
    padding = 0  # Changed padding to 0 as it was causing issues
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - different batch size
    input_tensor = np.array([[[[[1.0]]]], [[[[2.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0]]]], [[[[0]]]]], dtype=np.int64)
    output_size = (2, 1, 1, 1, 1)
    stride = 1
    padding = 0
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - different channel size
    input_tensor = np.array([[[[[1.0]], [[2.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0]], [[0]]]]], dtype=np.int64)
    output_size = (1, 2, 1, 1, 1)
    stride = 1
    padding = 0
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Different depth
    input_tensor = np.array([[[[[1.0], [2.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0], [1]]]]], dtype=np.int64)
    output_size = (1, 1, 1, 1, 2)
    stride = 1
    padding = 0
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - more complex example
    input_tensor = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0, 5], [10, 15]], [[20, 25], [30, 35]]]]], dtype=np.int64)
    output_size = (1, 1, 2, 8, 8)
    stride = 2
    padding = 0 #Changed padding
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_tensor = np.array([[[[[1.0]]]]], dtype=np.float32)
    indices_tensor = np.array([[[[[0]]]]], dtype=np.int64)
    output_size = (1, 1, 1, 2, 2)
    stride = 1
    padding = 0
    input_dict = {"input": input_tensor, "indices": indices_tensor, "output_size": output_size, "stride": stride, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.max_unpool3d"] = max_unpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.max_unpool3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_unpool3d'.")

check_valid('torch.nn.functional.max_unpool3d', generated_inputs['torch.nn.functional.max_unpool3d'], lib="torch", suffix=0)
