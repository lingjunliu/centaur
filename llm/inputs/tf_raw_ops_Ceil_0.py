
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ceil_inputs():
    list_of_inputs = []

    # Input 1: half, positive
    x = np.array([1.5, 2.3, 3.7], dtype=np.float16)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: half, negative
    x = np.array([-1.5, -2.3, -3.7], dtype=np.float16)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, mixed positive and negative
    x = np.array([-1.5, 2.3, -3.7, 4.1], dtype=np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "name": "ceil_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, large values
    x = np.array([1000.5, 2000.3, 3000.7], dtype=np.float64)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, multi-dimensional array
    x = np.array([[1.5, 2.3], [3.7, 4.1]], dtype=np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half, zero values
    x = np.array([0.0, -0.0, 1.0], dtype=np.float16)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64, small values
    x = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float32, a larger array
    x = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float32).reshape((2,3))
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, with integer values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, a 3D array
    x = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64)
    input_dict = {"x": tf.constant(x).numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Ceil"] = tf_raw_ops_ceil_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Ceil'.")

check_valid('tf.raw_ops.Ceil', generated_inputs['tf.raw_ops.Ceil'], lib="tf", suffix=0)
