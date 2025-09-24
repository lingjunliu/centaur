
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_pow_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 0.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow_example_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer input
    x = np.array([2, 3, 4], dtype=np.int32)
    y = np.array([2, 3, 2], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "pow_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16
    x = np.array([2.0, 3.0, 4.0], dtype=np.float16)
    y = np.array([2.0, 3.0, 0.5], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "pow_example_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimensional array
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, 0.5], [3.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative exponents
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([-2.0, -1.0, -0.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow_example_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex numbers
    x = np.array([1+1j, 2+2j], dtype=np.complex64)
    y = np.array([2+0j, 0.5+0j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "pow_example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64
    x = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([2.0, 3.0, 0.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "pow_example_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int8
    x = np.array([2, 3, 4], dtype=np.int8)
    y = np.array([2, 3, 2], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name": "pow_example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int64
    x = np.array([2, 3, 4], dtype=np.int64)
    y = np.array([2, 3, 2], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "pow_example_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128
    x = np.array([1+1j, 2+2j], dtype=np.complex128)
    y = np.array([2+0j, 0.5+0j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": "pow_example_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int16
    x = np.array([2, 3, 4], dtype=np.int16)
    y = np.array([2, 3, 2], dtype=np.int16)
    input_dict = {"x": x, "y": y, "name": "pow_example_12"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Pow'.")

check_valid('tf.raw_ops.Pow', generated_inputs['tf.raw_ops.Pow'], lib="tf", suffix=0)
