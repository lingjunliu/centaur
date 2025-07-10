
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.float32(1.0)
    name = "bessel_i1e_1"
    input_dict = {"x": tf.constant(np.array([x]), dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    name = "bessel_i1e_2"
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "bessel_i1e_3"
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    x = np.float64(-2.0)
    name = "bessel_i1e_4"
    input_dict = {"x": tf.constant(np.array([x]), dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, 1D array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    name = "bessel_i1e_5"
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, scalar
    x = np.float16(0.5)
    name = "bessel_i1e_9"
    input_dict = {"x": tf.constant(np.array([x]), dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: half, 1D array
    x = np.array([-0.5, 0.5], dtype=np.float16)
    name = "bessel_i1e_10"
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: float32, zero value
    x = np.float32(0.0)
    name = "bessel_i1e_11"
    input_dict = {"x": tf.constant(np.array([x]), dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float64, larger value
    x = np.float64(10.0)
    name = "bessel_i1e_12"
    input_dict = {"x": tf.constant(np.array([x]), dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.bessel_i1e"] = tf_math_bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.bessel_i1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bessel_i1e'.")

check_valid('tf.math.bessel_i1e', generated_inputs['tf.math.bessel_i1e'], lib="tf", suffix=0)
