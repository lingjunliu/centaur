
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_xlog1py_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1.0j, 2.0j, 3.0j], dtype=np.complex64)
    y = np.array([0.5j, 1.0j, 1.5j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1.0+1j, 2.0+2j, 3.0+3j], dtype=np.complex128)
    y = np.array([0.5+0.5j, 1.0+1j, 1.5+1.5j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[-0.9, -0.5], [0.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Xlog1py"] = tf_raw_ops_xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Xlog1py' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Xlog1py'.")

check_valid('tf.raw_ops.Xlog1py', generated_inputs['tf.raw_ops.Xlog1py'], lib="tf", suffix=0)
