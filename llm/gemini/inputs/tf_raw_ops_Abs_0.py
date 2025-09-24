
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_abs_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32
    x = np.float32(-5.0)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of float32
    x = np.array([-1.0, 0.0, 2.0, -3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "abs_1d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of float64
    x = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float64)
    input_dict = {"x": x, "name": "abs_2d_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array of int32
    x = np.array([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]], dtype=np.int32)
    input_dict = {"x": x, "name": "abs_3d_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar int8
    x = np.int8(-100)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of int64
    x = np.array([-1, 0, 2, -3], dtype=np.int64)
    input_dict = {"x": x, "name": "abs_1d_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of half (float16)
    x = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float16)
    input_dict = {"x": x, "name": "abs_2d_float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar int16
    x = np.int16(-32000)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array of float32
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x, "name": "abs_empty_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All positive values
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"x": x, "name": "abs_all_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Abs"] = tf_raw_ops_abs_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Abs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Abs'.")

check_valid('tf.raw_ops.Abs', generated_inputs['tf.raw_ops.Abs'], lib="tf", suffix=0)
