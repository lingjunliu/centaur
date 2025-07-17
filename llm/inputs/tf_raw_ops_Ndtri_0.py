
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ndtri_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": "ndtri_float32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float16, 2D array
    x = np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": "ndtri_float16_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 3D array
    x = np.array([[[0.3, 0.7], [0.5, 0.9]], [[0.1, 0.4], [0.6, 0.8]]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": "ndtri_float32_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    x = np.array(0.75, dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "ndtri_float64_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float16, 2D array with values close to 0 and close to 1
    x = np.array([[0.0001, 0.9999], [0.5, 0.1]], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": "ndtri_float16_2d_extreme"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16, empty array
    x = np.array([], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": "ndtri_float16_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, large values
    x = np.array([0.99999, 0.999999, 0.9999999], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": "ndtri_float32_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, values around 0.5
    x = np.array([0.49, 0.5, 0.51], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "ndtri_float64_around_05"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, 4D array
    x = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": "ndtri_float32_4d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16, 1D array with different values
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": "ndtri_float16_1d_diff"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Ndtri"] = tf_raw_ops_ndtri_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Ndtri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Ndtri'.")

check_valid('tf.raw_ops.Ndtri', generated_inputs['tf.raw_ops.Ndtri'], lib="tf", suffix=0)
