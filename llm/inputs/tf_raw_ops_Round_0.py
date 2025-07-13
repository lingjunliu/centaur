
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_round_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([1.0, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[1.1, 2.5], [3.4, 4.6]], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "round_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, 1D array with negative values
    x = np.array([-1.0, -1.2, -1.5, -1.8, -2.0], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.bfloat16), "name": "round_example_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 2D array with mixed values
    x = np.array([[-1.1, 2.5], [3.4, -4.6]], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D array
    x = np.array([[[1.1, 2.5], [3.4, 4.6]], [[5.7, 6.8], [7.9, 8.0]]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": "round_example_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: float64, scalar
    x = np.array(3.14159, dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32, 1D array. This should technically be a no-op
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "name": "round_example_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64
    x = np.array([1.0 + 1.0j, 2.0 - 2.0j], dtype=np.complex64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex64), "name": "round_example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128, 2D array
    x = np.array([[1.1 + 2.2j, 3.3 - 4.4j], [5.5 + 6.6j, 7.7 - 8.8j]], dtype=np.complex128)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex128), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64, scalar
    x = np.array(-10, dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int64), "name": "round_example_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Round"] = tf_raw_ops_round_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Round'.")

check_valid('tf.raw_ops.Round', generated_inputs['tf.raw_ops.Round'], lib="tf", suffix=0)
