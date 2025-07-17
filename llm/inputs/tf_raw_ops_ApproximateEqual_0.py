
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_approximate_equal_inputs():
    list_of_inputs = []

    # Input 1: float32, simple case
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.01, 1.99, 3.0], dtype=np.float32)
    tolerance = 0.05
    name = "approx_equal_1"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different tolerance
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0001, 2.0, 3.00001], dtype=np.float64)
    tolerance = 0.000001
    name = "approx_equal_2"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, integer values - tolerance must be float
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([1, 2, 3], dtype=np.float32)
    tolerance = 0.0
    name = "approx_equal_3"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, unsigned integer
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([1, 2, 4], dtype=np.float32)
    tolerance = 1.0
    name = "approx_equal_4"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, signed integer
    x = np.array([-1, 2, -3], dtype=np.float32)
    y = np.array([-1, 2, -2], dtype=np.float32)
    tolerance = 1.5
    name = "approx_equal_5"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, small integers
    x = np.array([-1, 0, 1], dtype=np.float32)
    y = np.array([-1, 0, 1], dtype=np.float32)
    tolerance = 0.0
    name = "approx_equal_6"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, large integers
    x = np.array([1000000000.0, 2000000000.0], dtype=np.float64)
    y = np.array([1000000001.0, 2000000000.0], dtype=np.float64)
    tolerance = 1.0
    name = "approx_equal_8"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[1.01, 1.99], [3.0, 4.01]], dtype=np.float32)
    tolerance = 0.05
    name = "approx_equal_10"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int64, cast to float
    x = np.array([1, 2, 3], dtype=np.int64).astype(np.float32)
    y = np.array([1, 2, 4], dtype=np.int64).astype(np.float32)
    tolerance = 1.0
    name = "approx_equal_11"
    input_dict = {"x": x, "y": y, "tolerance": tolerance, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApproximateEqual"] = tf_raw_ops_approximate_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApproximateEqual' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApproximateEqual'.")

check_valid('tf.raw_ops.ApproximateEqual', generated_inputs['tf.raw_ops.ApproximateEqual'], lib="tf", suffix=0)
