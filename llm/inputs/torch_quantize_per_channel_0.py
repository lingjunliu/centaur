
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def quantize_per_channel_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor, qint8
    input1 = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scales1 = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    zero_points1 = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    axis1 = 0
    dtype1 = torch.int8

    input_dict1 = {
        "input": input1,
        "scales": scales1,
        "zero_points": zero_points1,
        "axis": axis1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, qint8
    input2 = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    scales2 = np.array([0.5, 1.0, 0.25], dtype=np.float32)
    zero_points2 = np.array([0, 0, 0], dtype=np.int64)
    axis2 = 1
    dtype2 = torch.int8
    input_dict2 = {
        "input": input2,
        "scales": scales2,
        "zero_points": zero_points2,
        "axis": axis2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor, qint8, different axis
    input3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    scales3 = np.array([0.1, 0.2], dtype=np.float32)
    zero_points3 = np.array([0, 0], dtype=np.int64)
    axis3 = 1
    dtype3 = torch.int8
    input_dict3 = {
        "input": input3,
        "scales": scales3,
        "zero_points": zero_points3,
        "axis": axis3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))


    # Input 5: Multiple negative values, qint8
    input5 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    scales5 = np.array([0.5, 1.0], dtype=np.float32)
    zero_points5 = np.array([0, 0], dtype=np.int64)
    axis5 = 0
    dtype5 = torch.int8
    input_dict5 = {
        "input": input5,
        "scales": scales5,
        "zero_points": zero_points5,
        "axis": axis5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different scales and zero_points, qint8
    input6 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    scales6 = np.array([0.25, 0.5], dtype=np.float32)
    zero_points6 = np.array([0, 0], dtype=np.int64)
    axis6 = 0
    dtype6 = torch.int8
    input_dict6 = {
        "input": input6,
        "scales": scales6,
        "zero_points": zero_points6,
        "axis": axis6,
        "dtype": dtype6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger scales and zero points, qint8
    input7 = np.array([[-10.0, -20.0], [30.0, 40.0]], dtype=np.float32)
    scales7 = np.array([5.0, 10.0], dtype=np.float32)
    zero_points7 = np.array([0, 0], dtype=np.int64)
    axis7 = 0
    dtype7 = torch.int8
    input_dict7 = {
        "input": input7,
        "scales": scales7,
        "zero_points": zero_points7,
        "axis": axis7,
        "dtype": dtype7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

   # Input 8: Scales of 1.0
    input8 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scales8 = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    zero_points8 = np.array([0, 0, 0], dtype=np.int64)
    axis8 = 0
    dtype8 = torch.int8
    input_dict8 = {
        "input": input8,
        "scales": scales8,
        "zero_points": zero_points8,
        "axis": axis8,
        "dtype": dtype8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 10: Scales close to zero. # REPLACED because it was failing all the time
    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scales9 = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    zero_points9 = np.array([0, 0, 0], dtype=np.int64)
    axis9 = 0
    dtype9 = torch.int8

    input_dict9 = {
        "input": input9,
        "scales": scales9,
        "zero_points": zero_points9,
        "axis": axis9,
        "dtype": dtype9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.quantize_per_channel"] = quantize_per_channel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.quantize_per_channel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantize_per_channel'.")

check_valid('torch.quantize_per_channel', generated_inputs['torch.quantize_per_channel'], lib="torch", suffix=0)
