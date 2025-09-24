
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_igamma_inputs():
    list_of_inputs = []

    # Input 1: float32, simple values
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, different values, same shape
    a = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    x = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, zeros
    a = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    x = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, different shape
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, larger values
    a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    x = np.array([5.0, 15.0, 25.0], dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, small values
    a = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    x = np.array([0.04, 0.05, 0.06], dtype=np.float64)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, multi-dimensional arrays
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    x = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, different values
    a = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float64, simple values, different values
    a = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    input_dict = {"a": a, "x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Igamma"] = tf_raw_ops_igamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Igamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Igamma'.")

check_valid('tf.raw_ops.Igamma', generated_inputs['tf.raw_ops.Igamma'], lib="tf", suffix=0)
