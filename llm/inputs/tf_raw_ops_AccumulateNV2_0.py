
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_tf_raw_ops_AccumulateNV2_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, 2D tensors
    inputs_1 = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    ]
    input_dict_1 = {
        'inputs': [arr.copy() for arr in inputs_1],
        'shape': list(inputs_1[0].shape),
        'name': 'float32_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic int32, 1D vectors with negative numbers
    inputs_2 = [
        np.array([1, -2, 3], dtype=np.int32),
        np.array([-4, 5, -6], dtype=np.int32),
        np.array([7, 8, 9], dtype=np.int32)
    ]
    input_dict_2 = {
        'inputs': [arr.copy() for arr in inputs_2],
        'shape': list(inputs_2[0].shape),
        'name': 'int32_1d_negatives'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single tensor in list, float64
    inputs_3 = [np.array([10.123, 20.456, 30.789], dtype=np.float64)]
    input_dict_3 = {
        'inputs': [arr.copy() for arr in inputs_3],
        'shape': list(inputs_3[0].shape),
        'name': 'single_tensor_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar tensors (0-D), int64
    inputs_4 = [np.array(100, dtype=np.int64), np.array(200, dtype=np.int64)]
    input_dict_4 = {
        'inputs': [arr.copy() for arr in inputs_4],
        'shape': list(inputs_4[0].shape),
        'name': 'scalar_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: uint8, 2x3 tensors
    inputs_5 = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    ]
    input_dict_5 = {
        'inputs': [arr.copy() for arr in inputs_5],
        'shape': list(inputs_5[0].shape),
        'name': 'uint8_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D Tensors, float16 (half)
    inputs_6 = [
        np.array([[[1], [2]], [[3], [4]]], dtype=np.float16),
        np.array([[[5], [6]], [[7], [8]]], dtype=np.float16)
    ]
    input_dict_6 = {
        'inputs': [arr.copy() for arr in inputs_6],
        'shape': list(inputs_6[0].shape),
        'name': 'float16_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Long list of tensors, int16
    inputs_7 = [
        np.array([10, 20], dtype=np.int16),
        np.array([30, 40], dtype=np.int16),
        np.array([50, 60], dtype=np.int16),
        np.array([70, 80], dtype=np.int16)
    ]
    input_dict_7 = {
        'inputs': [arr.copy() for arr in inputs_7],
        'shape': list(inputs_7[0].shape),
        'name': 'long_list_int16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Complex numbers, complex64
    inputs_8 = [
        np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
        np.array([5 + 6j, 7 + 8j], dtype=np.complex64)
    ]
    input_dict_8 = {
        'inputs': [arr.copy() for arr in inputs_8],
        'shape': list(inputs_8[0].shape),
        'name': 'complex64_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensors with a zero dimension, uint32
    inputs_9 = [np.zeros((0, 5), dtype=np.uint32), np.zeros((0, 5), dtype=np.uint32)]
    input_dict_9 = {
        'inputs': [arr.copy() for arr in inputs_9],
        'shape': list(inputs_9[0].shape),
        'name': 'uint32_empty_dim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: int8 with mixed signs
    inputs_10 = [
        np.array([-10, 20, -30, 127], dtype=np.int8),
        np.array([40, -50, 60, -128], dtype=np.int8)
    ]
    input_dict_10 = {
        'inputs': [arr.copy() for arr in inputs_10],
        'shape': list(inputs_10[0].shape),
        'name': 'int8_mixed_sign'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: complex128
    inputs_11 = [
        np.array([[1.1e10 + 2.2e10j]], dtype=np.complex128),
        np.array([[3.3e10 + 4.4e10j]], dtype=np.complex128)
    ]
    input_dict_11 = {
        'inputs': [arr.copy() for arr in inputs_11],
        'shape': list(inputs_11[0].shape),
        'name': 'complex128_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: uint64 with large values
    inputs_12 = [
        np.array([2**63, 2**63 - 1], dtype=np.uint64),
        np.array([1, 1], dtype=np.uint64)
    ]
    input_dict_12 = {
        'inputs': [arr.copy() for arr in inputs_12],
        'shape': list(inputs_12[0].shape),
        'name': 'uint64_large_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.AccumulateNV2"] = generate_tf_raw_ops_AccumulateNV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.AccumulateNV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulateNV2'.")

check_valid('tf.raw_ops.AccumulateNV2', generated_inputs['tf.raw_ops.AccumulateNV2'], lib="tf", suffix=0)
