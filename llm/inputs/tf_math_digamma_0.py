
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_digamma_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "digamma_1"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    name = "digamma_4"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    name = "digamma_3"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "digamma_6"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1.0], dtype=np.float32)
    name = "digamma_7"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "digamma_8"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1e-2, 1e-1, 1.0], dtype=np.float32)
    name = "digamma_10"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[-0.5, -1.0], [0.0, 0.5]], dtype=np.float32)
    name = "digamma_11"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([-1.5, -0.5, 0.5], dtype=np.float64)
    name = "digamma_12"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1.2, 2.3, 3.4], dtype=np.float32)
    name = None
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.digamma"] = tf_math_digamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.digamma'.")

check_valid('tf.math.digamma', generated_inputs['tf.math.digamma'], lib="tf", suffix=0)
