
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DivNoNan_inputs():
    list_of_inputs = []

    # Input 1: Basic float division
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 0.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Division with complex numbers
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    y = np.array([1+0j, 0+0j, 3+0j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using float64
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([1.0, 0.0, 3.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    y = np.array([1+0j, 0+0j, 3+0j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Division with negative values
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    y = np.array([1.0, -0.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Division with zero and one values
    x = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    y = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shaped arrays for broadcasting.
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DivNoNan"] = tf_raw_ops_DivNoNan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DivNoNan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DivNoNan'.")

check_valid('tf.raw_ops.DivNoNan', generated_inputs['tf.raw_ops.DivNoNan'], lib="tf", suffix=0)
