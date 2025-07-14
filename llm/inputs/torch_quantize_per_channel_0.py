
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def quantize_per_channel_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input1 = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    scales1 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    zero_points1 = np.array([0, 1, 2, 3], dtype=np.int64)
    axis1 = 0
    dtype1 = torch.quint8  # Use quint8 for quantized unsigned 8-bit integer

    input_dict1 = {
        "input": input1,
        "scales": scales1,
        "zero_points": zero_points1,
        "axis": axis1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, different scales and zero points
    input2 = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    scales2 = np.array([0.5, 1.0], dtype=np.float32)
    zero_points2 = np.array([0, 128], dtype=np.int64)
    axis2 = 1
    dtype2 = torch.quint8  # Use quint8 for quantized unsigned 8-bit integer

    input_dict2 = {
        "input": input2,
        "scales": scales2,
        "zero_points": zero_points2,
        "axis": axis2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    scales3 = np.random.rand(3).astype(np.float32)
    zero_points3 = np.random.randint(0, 255, size=(3)).astype(np.int64)
    axis3 = 1
    dtype3 = torch.quint8  # Use quint8 for quantized unsigned 8-bit integer

    input_dict3 = {
        "input": input3,
        "scales": scales3,
        "zero_points": zero_points3,
        "axis": axis3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 5: Different dtype for quantization
    input5 = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    scales5 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    zero_points5 = np.array([128, 129, 130, 131], dtype=np.int64)  # zero points for uint8 should be around 128
    axis5 = 0
    dtype5 = torch.quint8

    input_dict5 = {
        "input": input5,
        "scales": scales5,
        "zero_points": zero_points5,
        "axis": axis5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Larger zero points
    input6 = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    scales6 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    zero_points6 = np.array([200, 201, 202, 203], dtype=np.int64)  # Zero points above 255 are likely invalid
    axis6 = 0
    dtype6 = torch.quint8

    input_dict6 = {
        "input": input6,
        "scales": scales6,
        "zero_points": zero_points6,
        "axis": axis6,
        "dtype": dtype6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Multi-dimensional scales. Keeping this, but simplifying.
    input7 = np.random.rand(2, 3, 4).astype(np.float32)
    scales7 = np.random.rand(3).astype(np.float32)
    zero_points7 = np.random.randint(0, 255, size=(3)).astype(np.int64)
    axis7 = 1
    dtype7 = torch.quint8

    input_dict7 = {
        "input": input7,
        "scales": scales7,
        "zero_points": zero_points7,
        "axis": axis7,
        "dtype": dtype7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Different Axis
    input8 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    scales8 = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    zero_points8 = np.array([128, 128, 128], dtype=np.int64)
    axis8 = 1
    dtype8 = torch.quint8

    input_dict8 = {
        "input": input8,
        "scales": scales8,
        "zero_points": zero_points8,
        "axis": axis8,
        "dtype": dtype8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 4D tensor
    input9 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    scales9 = np.random.rand(3).astype(np.float32)
    zero_points9 = np.random.randint(0, 255, size=3).astype(np.int64)
    axis9 = 1
    dtype9 = torch.quint8

    input_dict9 = {
        "input": input9,
        "scales": scales9,
        "zero_points": zero_points9,
        "axis": axis9,
        "dtype": dtype9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Large values, to test saturation
    input10 = np.array([-1000.0, 0.0, 1000.0, 2000.0], dtype=np.float32)
    scales10 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    zero_points10 = np.array([128, 129, 130, 131], dtype=np.int64)
    axis10 = 0
    dtype10 = torch.quint8

    input_dict10 = {
        "input": input10,
        "scales": scales10,
        "zero_points": zero_points10,
        "axis": axis10,
        "dtype": dtype10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Different shapes for scales and zero_points consistent with axis
    input11 = np.random.rand(2, 3, 4).astype(np.float32)
    scales11 = np.random.rand(2).astype(np.float32)
    zero_points11 = np.random.randint(0, 255, size=2).astype(np.int64)
    axis11 = 0
    dtype11 = torch.quint8

    input_dict11 = {
        "input": input11,
        "scales": scales11,
        "zero_points": zero_points11,
        "axis": axis11,
        "dtype": dtype11
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))

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
