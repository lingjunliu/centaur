
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_bessel_k1e_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    x = tf.constant(np.array([0.5, 1.0, 2.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Higher values
    x = tf.constant(np.array([5.0, 10.0, 20.0], dtype=np.float32))
    name = "bessel_k1e_test_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero value
    x = tf.constant(np.array([0.0], dtype=np.float32))
    name = "bessel_k1e_test_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of float64
    x = tf.constant(np.array([0.5, 1.0, 2.0], dtype=np.float64))
    name = "bessel_k1e_test_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of float32
    x = tf.constant(np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32))
    name = "bessel_k1e_test_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of float64
    x = tf.constant(np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float64))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large value
    x = tf.constant(np.array([100.0], dtype=np.float32))
    name = "bessel_k1e_test_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half type
    x = tf.constant(np.array([0.5, 1.0], dtype=np.float16))
    name = "bessel_k1e_test_10"
    x = tf.cast(x, dtype=tf.float16)
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex input with various values
    x = tf.constant(np.array([0.1, 1.5, 3.2, 7.8, 12.5], dtype=np.float32))
    name = "bessel_k1e_test_11"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Input with a specific shape
    x = tf.constant(np.reshape(np.array(range(1,7), dtype=np.float32), (2,3)))
    name = "bessel_k1e_test_12"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_math_special_bessel_k1e_inputs()
for i in range(len(inputs)):
    inputs[i]['x'] = inputs[i]['x'].numpy()
generated_inputs["tf.math.special.bessel_k1e"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.special.bessel_k1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_k1e'.")

check_valid('tf.math.special.bessel_k1e', generated_inputs['tf.math.special.bessel_k1e'], lib="tf", suffix=0)
