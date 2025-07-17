
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def layer_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D input
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    normalized_shape1 = (1,)
    weight1 = np.array([1.0], dtype=np.float32)
    bias1 = np.array([0.0], dtype=np.float32)
    eps1 = 1e-5
    input_dict1 = {
        "input": input1,
        "normalized_shape": normalized_shape1,
        "weight": weight1,
        "bias": bias1,
        "eps": eps1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D input
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    normalized_shape2 = (3,)
    weight2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    bias2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    eps2 = 1e-8
    input_dict2 = {
        "input": input2,
        "normalized_shape": normalized_shape2,
        "weight": weight2,
        "bias": bias2,
        "eps": eps2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D input
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    normalized_shape3 = (4,)
    weight3 = np.ones(4, dtype=np.float32)
    bias3 = np.zeros(4, dtype=np.float32)
    eps3 = 1e-6
    input_dict3 = {
        "input": input3,
        "normalized_shape": normalized_shape3,
        "weight": weight3,
        "bias": bias3,
        "eps": eps3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Input with negative values
    input4 = np.array([-1.0, -2.0, 3.0], dtype=np.float32)
    normalized_shape4 = (1,)
    weight4 = np.array([2.0], dtype=np.float32)
    bias4 = np.array([-1.0], dtype=np.float32)
    eps4 = 1e-5
    input_dict4 = {
        "input": input4,
        "normalized_shape": normalized_shape4,
        "weight": weight4,
        "bias": bias4,
        "eps": eps4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger eps
    input5 = np.random.rand(5, 5).astype(np.float32)
    normalized_shape5 = (5,)
    weight5 = np.ones(5, dtype=np.float32)
    bias5 = np.zeros(5, dtype=np.float32)
    eps5 = 0.1
    input_dict5 = {
        "input": input5,
        "normalized_shape": normalized_shape5,
        "weight": weight5,
        "bias": bias5,
        "eps": eps5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float64 input
    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    normalized_shape6 = (1,)
    weight6 = np.array([1.0], dtype=np.float64)
    bias6 = np.array([0.0], dtype=np.float64)
    eps6 = 1e-5
    input_dict6 = {
        "input": input6,
        "normalized_shape": normalized_shape6,
        "weight": weight6,
        "bias": bias6,
        "eps": eps6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: normalized_shape covering multiple dimensions
    input7 = np.random.rand(2, 3, 4).astype(np.float32)
    normalized_shape7 = (3,4)
    weight7 = np.ones((3,4), dtype=np.float32)
    bias7 = np.zeros((3,4), dtype=np.float32)
    eps7 = 1e-6
    input_dict7 = {
        "input": input7,
        "normalized_shape": normalized_shape7,
        "weight": weight7,
        "bias": bias7,
        "eps": eps7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: normalized_shape matching input shape.
    input8 = np.random.rand(2, 3, 4).astype(np.float32)
    normalized_shape8 = (2, 3, 4)
    weight8 = np.ones((2, 3, 4), dtype=np.float32)
    bias8 = np.zeros((2, 3, 4), dtype=np.float32)
    eps8 = 1e-6
    input_dict8 = {
        "input": input8,
        "normalized_shape": normalized_shape8,
        "weight": weight8,
        "bias": bias8,
        "eps": eps8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Different shapes and values.
    input9 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    normalized_shape9 = (2,)
    weight9 = np.array([0.5, 1.5], dtype=np.float32)
    bias9 = np.array([-0.5, 0.5], dtype=np.float32)
    eps9 = 1e-5

    input_dict9 = {
        "input": input9,
        "normalized_shape": normalized_shape9,
        "weight": weight9,
        "bias": bias9,
        "eps": eps9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Larger input, different scale.
    input10 = np.random.rand(10, 10).astype(np.float32) * 100
    normalized_shape10 = (10,)
    weight10 = np.ones(10, dtype=np.float32) * 0.1
    bias10 = np.zeros(10, dtype=np.float32) - 5
    eps10 = 1e-8

    input_dict10 = {
        "input": input10,
        "normalized_shape": normalized_shape10,
        "weight": weight10,
        "bias": bias10,
        "eps": eps10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: 4D input
    input11 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    normalized_shape11 = (5,)
    weight11 = np.ones(5, dtype=np.float32)
    bias11 = np.zeros(5, dtype=np.float32)
    eps11 = 1e-6
    input_dict11 = {
        "input": input11,
        "normalized_shape": normalized_shape11,
        "weight": weight11,
        "bias": bias11,
        "eps": eps11
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12: Remove weight and bias
    input12 = np.random.rand(5, 5).astype(np.float32)
    normalized_shape12 = (5,)
    eps12 = 0.1
    input_dict12 = {
        "input": input12,
        "normalized_shape": normalized_shape12,
        "weight": None,
        "bias": None,
        "eps": eps12
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.layer_norm"] = layer_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.layer_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.layer_norm'.")

check_valid('torch.nn.functional.layer_norm', generated_inputs['torch.nn.functional.layer_norm'], lib="torch", suffix=0)
