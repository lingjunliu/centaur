
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_abs_inputs():
    list_of_inputs = []

    # Input 1: bfloat16, scalar
    x = np.array(-5, dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: half, 1D array
    x = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float16)
    input_dict = {"x": x, "name": "abs_half_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[-1.5, 2.5], [-3.5, 4.5]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 3D array
    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "abs_float64_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int8, scalar
    x = np.array(-10, dtype=np.int8)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int16, 1D array
    x = np.array([-100, 200, -300, 400], dtype=np.int16)
    input_dict = {"x": x, "name": "abs_int16_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32, 2D array
    x = np.array([[-1000, 2000], [-3000, 4000]], dtype=np.int32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64, 3D array
    x = np.array([[[100, -200], [300, -400]], [[-500, 600], [-700, 800]]], dtype=np.int64)
    input_dict = {"x": x, "name": "abs_int64_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, scalar zero
    x = np.array(0.0, dtype=np.float32)
    input_dict = {"x": x, "name": "abs_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32, larger negative value
    x = np.array(-2147483647, dtype=np.int32)
    input_dict = {"x": x, "name": "abs_large_negative"}
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
