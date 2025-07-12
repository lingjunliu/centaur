
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_abs_inputs():
    list_of_inputs = []

    # Input 1: int32, positive scalar
    x = np.array(5, dtype=np.int32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, negative scalar
    x = np.array(-5, dtype=np.int32)
    name = "negative_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 1D array
    x = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    name = "float32_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 2D array
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    name = "float64_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int16, 3D array
    x = np.array([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]], dtype=np.int16)
    name = "int16_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bfloat16, positive values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    name = "bfloat16_positive"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int8, negative values
    x = np.array([-10, -20, -30], dtype=np.int8)
    name = "int8_negative"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64, larger values
    x = np.array([-10000000000, 20000000000], dtype=np.int64)
    name = "int64_large"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half (float16), mixed values
    x = np.array([-1.5, 2.5, -3.5], dtype=np.float16)
    name = "half_mixed"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32, multi dimensional
    x = np.array([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]], [[9, -10], [11, -12]]], dtype=np.int32)
    name = "int32_multi_dimensional"
    input_dict = {"x": x, "name": name}
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
