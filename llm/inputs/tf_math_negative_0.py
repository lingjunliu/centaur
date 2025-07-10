
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_negative_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.float32(5.0)
    name = "neg_scalar_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 1D array
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float64)
    name = "neg_1d_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, 2D array
    x = np.array([[1, -2], [-3, 4]], dtype=np.int32)
    name = "neg_2d_int32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, 3D array
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    name = "neg_3d_int64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bfloat16, scalar
    x = np.array(3.14, dtype=np.float16)
    name = "neg_bfloat16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: complex64, 1D array
    x = np.array([1+1j, 2-2j, -3+0j], dtype=np.complex64)
    name = "neg_complex64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: complex128, scalar
    x = np.complex128(1j)
    name = "neg_complex128"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, empty array
    x = np.array([], dtype=np.float32)
    name = "neg_empty_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8, scalar
    x = np.int8(-127)
    name = "neg_int8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16, scalar
    x = np.int16(32767)
    name = "neg_int16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.negative"] = tf_math_negative_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.negative' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.negative'.")

check_valid('tf.math.negative', generated_inputs['tf.math.negative'], lib="tf", suffix=0)
