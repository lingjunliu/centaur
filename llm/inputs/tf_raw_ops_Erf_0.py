
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Erf_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[0.5, 1.5], [-0.5, -1.5]], dtype=np.float64)
    input_dict = {"x":  tf.constant(x), "name": "my_erf"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, 1D array with large values
    x = np.array([-10.0, 10.0, -20.0, 20.0], dtype=np.float16)
    input_dict = {"x":  tf.constant(x, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float16, scalar value
    x = np.array(0.75, dtype=np.float16)
    input_dict = {"x":  tf.constant(x, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D array with zeros
    x = np.zeros((2, 2, 2), dtype=np.float32)
    input_dict = {"x":  tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 1D array with fractional values
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    input_dict = {"x":  tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, 2D array with mixed values
    x = np.array([[-2.5, 1.5], [0.0, -0.5]], dtype=np.float16)
    input_dict = {"x":  tf.constant(x, dtype=tf.float16), "name": "erf_float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16, 1D array
    x = np.array([-1.5, -1.0, 0.0, 1.0, 1.5], dtype=np.float16)
    input_dict = {"x":  tf.constant(x, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, scalar 1.0
    x = np.array(1.0, dtype=np.float32)
    input_dict = {"x":  tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, 2D array all values equal
    x = np.full((3, 3), 0.6, dtype=np.float64)
    input_dict = {"x":  tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Erf"] = tf_raw_ops_Erf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erf'.")

check_valid('tf.raw_ops.Erf', generated_inputs['tf.raw_ops.Erf'], lib="tf", suffix=0)
