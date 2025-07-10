
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_asin_inputs():
    list_of_inputs = []

    # Input 1: float32, single value
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, multiple values
    x = np.array([-0.5, 0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x), "name": "asin_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, multiple values
    x = np.array([-1.0, -0.707, 0.0, 0.707, 1.0], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, single value
    x = np.complex64(0.5 + 0.5j)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128, multiple values
    x = np.array([0.0 + 0.0j, 0.5 + 0.5j, -0.5 - 0.5j], dtype=np.complex128)
    input_dict = {"x": tf.convert_to_tensor(x), "name": "complex_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, 2D array
    x = np.array([[-0.8, 0.2], [0.5, 0.9]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64, 3D array
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16 - convert to numpy array AND convert back to tf.bfloat16
    x = np.array([-0.25, 0.75], dtype=np.float32).astype(np.float16)
    input_dict = {"x": tf.cast(tf.convert_to_tensor(x), dtype=tf.bfloat16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half
    x = np.array([-0.25, 0.75], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, values close to the boundary
    x = np.array([-0.99, 0.99], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.asin"] = tf_math_asin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.asin'.")

check_valid('tf.math.asin', generated_inputs['tf.math.asin'], lib="tf", suffix=0)
