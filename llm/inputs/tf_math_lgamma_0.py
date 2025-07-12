
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_lgamma_inputs():
    list_of_inputs = []

    # Input 1: Basic positive float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x.tolist()), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic negative float32
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x.tolist()), "name": "negative_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed positive and negative float32
    x = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x.tolist()), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x.tolist()), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"x": tf.constant(x.tolist()), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"x": tf.constant(x.tolist()), "name": "half_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional float32
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": tf.constant(x.tolist()), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional float64
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"x": tf.constant(x.tolist()), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Values close to zero
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    input_dict = {"x": tf.constant(x.tolist()), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger negative values
    x = np.array([-5.0, -10.0, -15.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x.tolist()), "name": "large_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Single value
    x = np.array([5.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x.tolist()), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.lgamma"] = tf_math_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.lgamma'.")

check_valid('tf.math.lgamma', generated_inputs['tf.math.lgamma'], lib="tf", suffix=0)
