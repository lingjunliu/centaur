
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_realdiv_inputs():
    list_of_inputs = []

    # Input 1
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=tf.float32)
    y = tf.constant(np.array([0.5, 0.5, 0.5], dtype=np.float32), dtype=tf.float32)
    name = "real_div_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), dtype=tf.float64)
    y = tf.constant(np.array([[0.2, 0.5], [1.0, 2.0]], dtype=np.float64), dtype=tf.float64)
    name = "real_div_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = tf.constant(np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32), dtype=tf.float32)
    y = tf.constant(np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32), dtype=tf.float32)
    name = "real_div_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=tf.float32)
    y = tf.constant(np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=tf.float32)
    name = "real_div_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=tf.float32)
    y = tf.constant(np.array([1.0, 2.0, 0.5], dtype=np.float32), dtype=tf.float32)
    name = "real_div_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64), dtype=tf.float64)
    y = tf.constant(np.array([1.0, 1.0, 1.0], dtype=np.float64), dtype=tf.float64)
    name = "real_div_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64), dtype=tf.float64)
    y = tf.constant(np.array([1.0, 1.0, 1.0], dtype=np.float64), dtype=tf.float64)
    name = "real_div_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), dtype=tf.float32)
    y = tf.constant(np.array([0.5, 1.0], dtype=np.float32), dtype=tf.float32)
    name = "real_div_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = tf.constant(10.0, dtype=tf.float32)
    y = tf.constant(2.0, dtype=tf.float32)
    name = "real_div_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64), dtype=tf.float64)
    y = tf.constant(2.0, dtype=tf.float64)
    y = tf.constant(np.array([2.0,2.0,2.0], dtype = np.float64), dtype=tf.float64)
    name = "real_div_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.realdiv"] = tf_realdiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.realdiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.realdiv'.")

check_valid('tf.realdiv', generated_inputs['tf.realdiv'], lib="tf", suffix=0)
