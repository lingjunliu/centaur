
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_ndtri_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    name = "ndtri_1"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (double)
    x = np.array([0.2, 0.6, 0.8], dtype=np.float64)
    name = "ndtri_2"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Input close to 0
    x = np.array([0.0001, 0.001, 0.01], dtype=np.float32)
    name = "ndtri_3"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Input close to 1
    x = np.array([0.99, 0.999, 0.9999], dtype=np.float32)
    name = "ndtri_4"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single value
    x = np.array(0.5, dtype=np.float32)
    name = "ndtri_5"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional array
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    name = "ndtri_6"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty name
    x = np.array([0.4, 0.7], dtype=np.float32)
    name = ""
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Longer name
    x = np.array([0.25, 0.75], dtype=np.float32)
    name = "this_is_a_very_long_name"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger array
    x = np.random.rand(100).astype(np.float32)
    name = "ndtri_9"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with different values
    x = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], dtype=np.float32)
    name = "ndtri_10"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Three dimensional array
    x = np.random.rand(2, 3, 4).astype(np.float32)
    name = "ndtri_11"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Array with values close to 0.5
    x = np.array([0.49, 0.5, 0.51], dtype=np.float32)
    name = "ndtri_12"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.ndtri"] = tf_math_ndtri_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.ndtri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.ndtri'.")

check_valid('tf.math.ndtri', generated_inputs['tf.math.ndtri'], lib="tf", suffix=0)
