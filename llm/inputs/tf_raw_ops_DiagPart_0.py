
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_diag_part_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2
    input_1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict_1 = {"input": input_1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 4x4 float64
    input_2 = np.array([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0],
                         [13.0, 14.0, 15.0, 16.0]], dtype=np.float64)
    input_dict_2 = {"input": input_2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Complex64
    input_3 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict_3 = {"input": input_3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 6x6 int64
    input_4 = np.array([[i * 6 + j for j in range(6)] for i in range(6)], dtype=np.int64)
    input_dict_4 = {"input": input_4, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2x2 bfloat16 (requires casting from float32)
    input_5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).astype(np.float16)
    input_dict_5 = {"input": input_5, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 4x4 half (requires casting from float32)
    input_6 = np.array([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0],
                         [13.0, 14.0, 15.0, 16.0]], dtype=np.float32).astype(np.float16)
    input_dict_6 = {"input": input_6, "name": "diag_part_half"}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 2x2 complex128
    input_7 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict_7 = {"input": input_7, "name": "diag_part_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2x2 int32 with negative values
    input_8 = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    input_dict_8 = {"input": input_8, "name": "diag_part_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 2x2 float32 with negative values
    input_9 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict_9 = {"input": input_9, "name": "diag_part_negative_float"}
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: 2x2x2x2 int32
    input_10 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]] , dtype=np.int32)
    input_dict_10 = {"input": input_10, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DiagPart"] = tf_raw_ops_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DiagPart'.")

check_valid('tf.raw_ops.DiagPart', generated_inputs['tf.raw_ops.DiagPart'], lib="tf", suffix=0)
