
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_add_n_inputs():
    list_of_inputs = []

    def create_input_dict(inputs, name):
        return {"inputs": inputs, "name": name}

    # Input 1: Basic test with int32
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "add_basic_int32")))

    # Input 2: Test with float32
    inputs = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "add_basic_float32")))

    # Input 3: Test with negative values
    inputs = [np.array([[-1, 2], [3, -4]], dtype=np.int32), np.array([[5, -6], [-7, 8]], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "add_negative_int32")))

    # Input 4: Test with 3D tensors
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "add_3d_int32")))

    # Input 5: Test with an empty name
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "")))

    # Input 6: Test with a long name
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "this_is_a_very_long_name_for_the_operation")))

    # Input 7: Test with int64
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int64), np.array([[5, 6], [7, 8]], dtype=np.int64)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "add_basic_int64")))

    # Input 8: Test with float64
    inputs = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "add_basic_float64")))

    # Input 9: test with all zeros
    inputs = [np.zeros((2, 2), dtype=np.int32), np.zeros((2, 2), dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs, "add_zeros")))

    return list_of_inputs

generated_inputs["tf.math.add_n"] = tf_math_add_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.add_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.add_n'.")

check_valid('tf.math.add_n', generated_inputs['tf.math.add_n'], lib="tf", suffix=0)
