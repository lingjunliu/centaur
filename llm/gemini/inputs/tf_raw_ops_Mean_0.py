
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_mean_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 with axis=0
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis1 = np.array(0, dtype=np.int32)
    input_dict1 = {"input": input1, "axis": axis1, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: float64 with axis=1 and keep_dims=True
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    axis2 = np.array(1, dtype=np.int64)
    input_dict2 = {"input": input2, "axis": axis2, "keep_dims": True, "name": "mean_example"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: int32 with axis=[0, 1]
    input3 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis3 = np.array([0, 1], dtype=np.int32)
    input_dict3 = {"input": input3, "axis": axis3, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: complex64 with negative axis
    input4 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    axis4 = np.array(-1, dtype=np.int32)
    input_dict4 = {"input": input4, "axis": axis4, "keep_dims": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: int64, 3 dimensions
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axis5 = np.array(0, dtype=np.int64)
    input_dict5 = {"input": input5, "axis": axis5, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: uint8
    input6 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    axis6 = np.array(1, dtype=np.int32)
    input_dict6 = {"input": input6, "axis": axis6, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: half
    input7 = np.array([[1, 2], [3, 4]], dtype=np.float16)
    axis7 = np.array(0, dtype=np.int32)
    input_dict7 = {"input": input7, "axis": axis7, "keep_dims": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Two axis
    input8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis8 = np.array([0, 2], dtype=np.int32)
    input_dict8 = {"input": input8, "axis": axis8, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: complex128
    input9 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    axis9 = np.array(0, dtype=np.int64)
    input_dict9 = {"input": input9, "axis": axis9, "keep_dims": True, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: int16
    input10 = np.array([[1, 2], [3, 4]], dtype=np.int16)
    axis10 = np.array(1, dtype=np.int32)
    input_dict10 = {"input": input10, "axis": axis10, "keep_dims": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Mean"] = tf_raw_ops_mean_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Mean' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Mean'.")

check_valid('tf.raw_ops.Mean', generated_inputs['tf.raw_ops.Mean'], lib="tf", suffix=0)
