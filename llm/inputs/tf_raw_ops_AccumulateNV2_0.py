
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulate_nv2_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensors
    inputs = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    shape = [3]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_float32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 tensors
    inputs = [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)]
    shape = [3]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_int32_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 tensors
    inputs = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)]
    shape = [2, 2]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_float64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values, int64
    inputs = [np.array([-1, -2, -3], dtype=np.int64), np.array([-4, -5, -6], dtype=np.int64)]
    shape = [3]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_int64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More tensors
    inputs = [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32), np.array([7, 8, 9], dtype=np.int32)]
    shape = [3]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_int32_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bfloat16 tensors
    inputs = [np.array([1.0, 2.0], dtype=np.float16), np.array([3.0, 4.0], dtype=np.float16)]
    shape = [2]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_bfloat16_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64
    inputs = [np.array([1+1j, 2+2j], dtype=np.complex64), np.array([3+3j, 4+4j], dtype=np.complex64)]
    shape = [2]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_complex64_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8
    inputs = [np.array([1, 2, 3], dtype=np.uint8), np.array([4, 5, 6], dtype=np.uint8)]
    shape = [3]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_uint8_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int32
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)]
    shape = [2, 2, 2]
    input_dict = {"inputs": [tf.convert_to_tensor(x) for x in inputs], "shape": shape, "name": "accumulate_int32_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty arrays
    inputs = [np.array([], dtype=np.float32), np.array([], dtype=np.float32)]
    shape = [0]
    input_dict = {"inputs": [tf.convert_to_tensor(x, dtype=tf.float32) for x in inputs], "shape": shape, "name": "accumulate_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.AccumulateNV2"] = tf_raw_ops_accumulate_nv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AccumulateNV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AccumulateNV2'.")

check_valid('tf.raw_ops.AccumulateNV2', generated_inputs['tf.raw_ops.AccumulateNV2'], lib="tf", suffix=0)
