
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_accumulate_nv2_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    inputs = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
              np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int32),
              np.array([[5, 6], [7, 8]], dtype=np.int32)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64
    inputs = [np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64),
              np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex64)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 with different shape
    inputs = [np.array([1, 2, 3, 4, 5], dtype=np.int64),
              np.array([6, 7, 8, 9, 10], dtype=np.int64)]
    shape = [5]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 with 3D tensor
    inputs = [np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64),
              np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float64)]
    shape = [2, 2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.uint8),
              np.array([[5, 6], [7, 8]], dtype=np.uint8)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  int32
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int32)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16
    inputs = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16).astype(np.float16),
              np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float16).astype(np.float16)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.uint32),
              np.array([[5, 6], [7, 8]], dtype=np.uint32)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128
    inputs = [np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128),
              np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex128)]
    shape = [2, 2]
    input_dict = {"inputs": inputs, "shape": shape}
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
