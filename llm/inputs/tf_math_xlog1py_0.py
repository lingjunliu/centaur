
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_xlog1py_inputs():
    list_of_inputs = []

    # Input 1
    x = tf.constant(0.0, dtype=tf.float32).numpy()
    y = tf.constant(1.0, dtype=tf.float32).numpy()
    name = "input1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = tf.constant(1.0, dtype=tf.float32).numpy()
    y = tf.constant(1.0, dtype=tf.float32).numpy()
    name = "input2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = tf.constant(2.0, dtype=tf.float32).numpy()
    y = tf.constant(2.0, dtype=tf.float32).numpy()
    name = "input3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = tf.constant(0.0, dtype=tf.float32).numpy()
    y = tf.constant(-0.5, dtype=tf.float32).numpy()
    name = "input4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = tf.constant(-1.0, dtype=tf.float32).numpy()
    y = tf.constant(0.5, dtype=tf.float32).numpy()
    name = "input5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    y = tf.constant([0.1, 0.2, 0.3], dtype=tf.float32).numpy()
    name = "input6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy()
    y = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32).numpy()
    name = "input7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = tf.constant(1.0, dtype=tf.float64).numpy()
    y = tf.constant(1.0, dtype=tf.float64).numpy()
    name = "input8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = (1.0 + 1j) * np.ones((), dtype=np.complex64)
    y = (1.0 + 1j) * np.ones((), dtype=np.complex64)
    name = "input9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = tf.constant(0.0, dtype=tf.float32).numpy()
    y = tf.constant(-0.99, dtype=tf.float32).numpy()
    name = "input10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.xlog1py"] = tf_math_xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.xlog1py' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.xlog1py'.")

check_valid('tf.math.xlog1py', generated_inputs['tf.math.xlog1py'], lib="tf", suffix=0)
