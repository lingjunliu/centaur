
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_square_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    name = "square_float32_1d"
    input_dict = {"x": tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, 2D array
    x = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    name = "square_int32_2d"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    name = "square_float64_3d"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64, scalar
    x = np.array(-5, dtype=np.int64)
    name = "square_int64_scalar"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8, 1D array
    x = np.array([1, 2, 3], dtype=np.uint8)
    name = "square_uint8_1d"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, 2D array
    x = np.array([[1+1j, 2-2j], [3+0j, 4-1j]], dtype=np.complex64)
    name = "square_complex64_2d"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16, 1D array
    x = np.array([-1.5, 0.0, 2.5], dtype=np.float16).astype(np.float32) #tf.bfloat16 only works with eager mode which is not used here.
    name = "square_bfloat16_1d"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, 2D array
    x = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float16) #tf.float16 is half
    name = "square_half_2d"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int16, 1D array
    x = np.array([-100, 0, 100], dtype=np.int16)
    name = "square_int16_1d"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, 1D array
    x = np.array([1+1j, 2-2j], dtype=np.complex128)
    name = "square_complex128_1d"
    input_dict = {"x":  tf.constant(x), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Square"] = tf_raw_ops_square_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Square'.")

check_valid('tf.raw_ops.Square', generated_inputs['tf.raw_ops.Square'], lib="tf", suffix=0)
