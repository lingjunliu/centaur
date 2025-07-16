
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_approximate_equal_inputs():
    list_of_inputs = []

    # Input 1: float32, basic test
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.01, 1.99, 3.0], dtype=np.float32)
    tolerance = 0.05
    name = "approx_equal_1"
    input_dict = {"x": x, "y": y, "tolerance": np.float32(tolerance), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different tolerance
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.01, 1.99, 3.0], dtype=np.float64)
    tolerance = 0.001
    name = "approx_equal_2"
    input_dict = {"x": x, "y": y, "tolerance": np.float64(tolerance), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8, Removing this due to errors
    # x = np.array([1, 2, 3], dtype=np.uint8)
    # y = np.array([1, 2, 3], dtype=np.uint8)
    # tolerance = 0.0
    # name = "approx_equal_4"
    # input_dict = {"x": x, "y": y, "tolerance": np.float32(tolerance), "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int16, negative values
    x = np.array([-1, -2, -3], dtype=np.int16)
    y = np.array([-1, -2, -3], dtype=np.int16)
    tolerance = 0.0
    name = "approx_equal_5"
    input_dict = {"x": x, "y": y, "tolerance": np.float32(tolerance), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[1.01, 1.99], [3.0, 4.0]], dtype=np.float32)
    tolerance = 0.05
    name = "approx_equal_6"
    input_dict = {"x": x, "y": y, "tolerance": np.float32(tolerance), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64
    x = np.array([1+1j, 2+2j], dtype=np.complex64)
    y = np.array([1.01+1.01j, 1.99+1.99j], dtype=np.complex64)
    tolerance = 0.05
    name = "approx_equal_7"
    input_dict = {"x": x, "y": y, "tolerance": np.float32(tolerance), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([1, 2, 4], dtype=np.int64)
    tolerance = 1.0
    name = "approx_equal_8"
    input_dict = {"x": x, "y": y, "tolerance": np.float64(tolerance), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float16)
    y = np.array([1.01, 1.99, 3.0], dtype=np.float32).astype(np.float16)
    tolerance = 0.05
    name = "approx_equal_9"
    input_dict = {"x": x, "y": y, "tolerance": np.float32(tolerance), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128
    x = np.array([1+1j, 2+2j], dtype=np.complex128)
    y = np.array([1.01+1.01j, 1.99+1.99j], dtype=np.complex128)
    tolerance = 0.05
    name = "approx_equal_10"
    input_dict = {"x": x, "y": y, "tolerance": np.float64(tolerance), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    #Input 11: half
    x = np.array([1.0, 2.0], dtype=np.float16)
    y = np.array([1.01, 1.99], dtype=np.float16)
    tolerance = 0.05
    name = "approx_equal_11"
    input_dict = {"x": x, "y": y, "tolerance": np.float32(tolerance), "name": name}
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
