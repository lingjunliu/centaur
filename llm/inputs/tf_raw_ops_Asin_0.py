
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_asin_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array, positive values
    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array, negative values
    x = np.array([-0.5, -1.0, 0.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x), "name": "asin_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 2D array
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 2D array, negative values
    x = np.array([[-0.2, -0.4], [-0.6, -0.8]], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x), "name": "asin_neg_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64, 1D array
    x = np.array([0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j], dtype=np.complex64)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128, 2D array
    x = np.array([[0.4 + 0.4j, 0.5 + 0.5j], [0.6 + 0.6j, 0.7 + 0.7j]], dtype=np.complex128)
    input_dict = {"x": tf.convert_to_tensor(x), "name": "asin_complex_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16, 1D array
    x = np.array([0.2, 0.8, 0.5], dtype=np.float16).astype(tf.bfloat16.as_numpy_dtype)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, 1D array
    x = np.array([-0.3, 0.7, -0.1], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x), "name": "asin_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, scalar value
    x = np.array(0.9, dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Asin"] = tf_raw_ops_asin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Asin'.")

check_valid('tf.raw_ops.Asin', generated_inputs['tf.raw_ops.Asin'], lib="tf", suffix=0)
