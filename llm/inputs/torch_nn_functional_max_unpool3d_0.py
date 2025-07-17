
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def max_unpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]]).astype(np.float32)
    indices1 = np.array([[[[[0, 1], [2, 3]]]]]).astype(np.int64)
    output_size1 = (1, 1, 4, 4, 2)
    stride1 = 2
    padding1 = 0
    input_dict1 = {"input": input1, "indices": indices1, "output_size": output_size1, "stride": stride1, "padding": padding1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[[[[1.0]]]]]).astype(np.float32)
    indices2 = np.array([[[[[0]]]]]).astype(np.int64)
    output_size2 = (1, 1, 2, 2, 1)
    stride2 = 2
    padding2 = 0
    input_dict2 = {"input": input2, "indices": indices2, "output_size": output_size2, "stride": stride2, "padding": padding2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]).astype(np.float32)
    indices3 = np.array([[[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]]]).astype(np.int64)
    output_size3 = (1, 1, 4, 4, 2)
    stride3 = 2
    padding3 = 0
    input_dict3 = {"input": input3, "indices": indices3, "output_size": output_size3, "stride": stride3, "padding": padding3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4 - Different stride
    input4 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]]).astype(np.float32)
    indices4 = np.array([[[[[0, 1], [2, 3]]]]]).astype(np.int64)
    output_size4 = (1, 1, 4, 4, 2)
    stride4 = 1
    padding4 = 0
    input_dict4 = {"input": input4, "indices": indices4, "output_size": output_size4, "stride": stride4, "padding": padding4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5 - Different padding
    input5 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]]).astype(np.float32)
    indices5 = np.array([[[[[0, 1], [2, 3]]]]]).astype(np.int64)
    output_size5 = (1, 1, 5, 5, 2)
    stride5 = 2
    padding5 = 1
    input_dict5 = {"input": input5, "indices": indices5, "output_size": output_size5, "stride": stride5, "padding": padding5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6 - Larger input size
    input6 = np.random.rand(1, 1, 2, 2, 2).astype(np.float32)
    indices6 = np.random.randint(0, 8, size=(1, 1, 2, 2, 2)).astype(np.int64)
    output_size6 = (1, 1, 4, 4, 4)
    stride6 = 2
    padding6 = 0
    input_dict6 = {"input": input6, "indices": indices6, "output_size": output_size6, "stride": stride6, "padding": padding6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7 - Negative Values
    input7 = np.array([[[[-1.0, 2.0], [-3.0, 4.0]]]]).astype(np.float32)
    indices7 = np.array([[[[[0, 1], [2, 3]]]]]).astype(np.int64)
    output_size7 = (1, 1, 4, 4, 2)
    stride7 = 2
    padding7 = 0
    input_dict7 = {"input": input7, "indices": indices7, "output_size": output_size7, "stride": stride7, "padding": padding7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8 - Batch Size 2
    input8 = np.random.rand(2, 1, 2, 2, 2).astype(np.float32)
    indices8 = np.random.randint(0, 8, size=(2, 1, 2, 2, 2)).astype(np.int64)
    output_size8 = (2, 1, 4, 4, 4)
    stride8 = 2
    padding8 = 0
    input_dict8 = {"input": input8, "indices": indices8, "output_size": output_size8, "stride": stride8, "padding": padding8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9 - Non-square input
    input9 = np.random.rand(1, 1, 2, 1, 3).astype(np.float32)
    indices9 = np.random.randint(0, 8, size=(1, 1, 2, 1, 3)).astype(np.int64)
    output_size9 = (1, 1, 4, 2, 6)
    stride9 = 2
    padding9 = 0
    input_dict9 = {"input": input9, "indices": indices9, "output_size": output_size9, "stride": stride9, "padding": padding9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10 - stride=1 padding=0
    input10 = np.random.rand(1, 1, 2, 2, 2).astype(np.float32)
    indices10 = np.random.randint(0, 8, size=(1, 1, 2, 2, 2)).astype(np.int64)
    output_size10 = (1, 1, 3, 3, 3)
    stride10 = 1
    padding10 = 0
    input_dict10 = {"input": input10, "indices": indices10, "output_size": output_size10, "stride": stride10, "padding": padding10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11 - stride=1 padding=1, output_size must accomodate padding and stride
    input11 = np.random.rand(1, 1, 2, 2, 2).astype(np.float32)
    indices11 = np.random.randint(0, 8, size=(1, 1, 2, 2, 2)).astype(np.int64)
    output_size11 = (1, 1, 5, 5, 5)  # Adjusted output_size
    stride11 = 1
    padding11 = 1
    input_dict11 = {"input": input11, "indices": indices11, "output_size": output_size11, "stride": stride11, "padding": padding11}
    list_of_inputs.append(copy.deepcopy(input_dict11))


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
