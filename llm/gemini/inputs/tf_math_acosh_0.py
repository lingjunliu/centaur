
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_acosh_inputs():
    list_of_inputs = []

    # Input 1: Valid input with float32
    x = tf.constant(np.array([1.0, 1.5, 2.0, 5.0]), dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid input with float64
    x = tf.constant(np.array([1.0, 1.1, 10.0, 100.0]), dtype=tf.float64)
    name = "acosh_example"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid input with half
    x = tf.constant(np.array([1.0, 1.2, 1.5, 2.0]), dtype=tf.float16)
    name = "another_acosh"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid input with complex64
    x = tf.constant(np.array([1+0j, 2+0j, 3+0j, 4+0j]), dtype=tf.complex64)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid input with complex128
    x = tf.constant(np.array([1+0j, 1.1+0j, 10+0j, 100+0j]), dtype=tf.complex128)
    name = "complex_acosh"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional array (float32)
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional array (float64)
    x = tf.constant(np.array([[1.1, 2.2], [3.3, 4.4]]), dtype=tf.float64)
    name = "multi_acosh"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Input with infinity (float32)
    x = tf.constant(np.array([1.0, np.inf, 2.0]), dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: larger values
    x = tf.constant(np.array([1.0, 1000.0, 100000.0]), dtype=tf.float32)
    name = "large_acosh"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = tf.constant(np.array([1.0]), dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    x = tf.constant(np.array([[1.0]]), dtype=tf.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
inputs = tf_math_acosh_inputs()
for i in range(len(inputs)):
    inputs[i]['x'] = inputs[i]['x'].numpy()
generated_inputs["tf.math.acosh"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.acosh'.")

check_valid('tf.math.acosh', generated_inputs['tf.math.acosh'], lib="tf", suffix=0)
