
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_betainc_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input with float32
    a = np.array([0.5], dtype=np.float32)
    b = np.array([0.5], dtype=np.float32)
    x = np.array([0.5], dtype=np.float32)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic valid input with float64
    a = np.array([0.5], dtype=np.float64)
    b = np.array([0.5], dtype=np.float64)
    x = np.array([0.5], dtype=np.float64)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different values, float32
    a = np.array([2.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)
    x = np.array([0.8], dtype=np.float32)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different values, float64
    a = np.array([2.0], dtype=np.float64)
    b = np.array([3.0], dtype=np.float64)
    x = np.array([0.8], dtype=np.float64)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional array, float32
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional array, float64
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x close to 0, float32
    a = np.array([1.0], dtype=np.float32)
    b = np.array([1.0], dtype=np.float32)
    x = np.array([0.0001], dtype=np.float32)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x close to 1, float64
    a = np.array([1.0], dtype=np.float64)
    b = np.array([1.0], dtype=np.float64)
    x = np.array([0.9999], dtype=np.float64)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger values for a and b, float32
    a = np.array([10.0], dtype=np.float32)
    b = np.array([15.0], dtype=np.float32)
    x = np.array([0.5], dtype=np.float32)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger values for a and b, float64
    a = np.array([10.0], dtype=np.float64)
    b = np.array([15.0], dtype=np.float64)
    x = np.array([0.5], dtype=np.float64)
    input_dict = {"name": None, "a": a, "b": b, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Betainc"] = tf_raw_ops_betainc_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Betainc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Betainc'.")

check_valid('tf.raw_ops.Betainc', generated_inputs['tf.raw_ops.Betainc'], lib="tf", suffix=0)
