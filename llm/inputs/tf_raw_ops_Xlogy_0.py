
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_xlogy_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "xlogy_basic_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 arrays, different values
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "xlogy_float64_diff_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex64 arrays
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    y = np.array([4 + 4j, 5 + 5j, 6 + 6j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "xlogy_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex128 arrays
    x = np.array([1 - 1j, 2 - 2j, 3 - 3j], dtype=np.complex128)
    y = np.array([4 - 4j, 5 - 5j, 6 - 6j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": "xlogy_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional float32 arrays
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "xlogy_multidimensional_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float16)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32).astype(np.float16)
    input_dict = {"x": x, "y": y, "name": "xlogy_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32 with zero in x
    x = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "xlogy_zero_in_x"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float32 with negative and positive values in x and y
    x = np.array([-1.0, 2.0, 0.0], dtype=np.float32)
    y = np.array([0.5, 5.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "xlogy_negative_and_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float32 with some very small numbers
    x = np.array([1e-8, 2e-8, 3e-8], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "xlogy_small_numbers"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different shaped arrays
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "xlogy_different_shapes"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Xlogy"] = tf_raw_ops_xlogy_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Xlogy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Xlogy'.")

check_valid('tf.raw_ops.Xlogy', generated_inputs['tf.raw_ops.Xlogy'], lib="tf", suffix=0)
