
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Xlog1py_inputs():
    list_of_inputs = []

    # Input 1: float32, positive values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    name = None

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([-4.0, -5.0, -6.0], dtype=np.float32)
    name = "negative_values"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, mixed values, multi-dimensional
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    y = np.array([[-5.0, 6.0], [-7.0, 8.0]], dtype=np.float64)
    name = "mixed_values_multi_dim"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64
    x = np.array([1 + 1j, 2 - 2j], dtype=np.complex64)
    y = np.array([3 - 1j, 4 + 2j], dtype=np.complex64)
    name = "complex64_values"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128
    x = np.array([1 + 1j, 2 - 2j], dtype=np.complex128)
    y = np.array([3 - 1j, 4 + 2j], dtype=np.complex128)
    name = "complex128_values"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, zero values in x
    x = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    name = "zero_in_x"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, very small values
    x = np.array([1e-8, 2e-8, 3e-8], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    name = "small_values"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float16)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32).astype(np.float16)
    name = "float16_values"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float16)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32).astype(np.float16)
    name = "half_values"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, y close to -1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([-0.9, -0.99, -0.999], dtype=np.float32)
    name = "y_close_to_minus_one"

    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Xlog1py"] = tf_raw_ops_Xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Xlog1py' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Xlog1py'.")

check_valid('tf.raw_ops.Xlog1py', generated_inputs['tf.raw_ops.Xlog1py'], lib="tf", suffix=0)
