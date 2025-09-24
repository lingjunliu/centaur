
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_polygamma_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    a = np.array(0.0, dtype=np.float32)
    x = np.array(1.0, dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": "polygamma_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic case with float64
    a = np.array(1.0, dtype=np.float64)
    x = np.array(2.0, dtype=np.float64)
    input_dict = {"a": a, "x": x, "name": "polygamma_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Array inputs, float64
    a = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"a": a, "x": x, "name": "polygamma_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional array inputs, float32
    a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": "polygamma_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large a and x values, float32
    a = np.array(10.0, dtype=np.float32)
    x = np.array(100.0, dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": "polygamma_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional with larger values, float64
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    x = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64)
    input_dict = {"a": a, "x": x, "name": "polygamma_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: a and x are 3d tensors, float64
    a = np.random.rand(2, 3, 4).astype(np.float64)
    x = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict = {"a": a, "x": x, "name": "polygamma_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting case, float32
    a = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array(2.0, dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": "polygamma_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar a and array x, float64
    a = np.array(2.0, dtype=np.float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"a": a, "x": x, "name": "polygamma_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative x with positive a.
    a = np.array(1.0, dtype=np.float32)
    x = np.array(-2.0, dtype=np.float32)
    input_dict = {"a": a, "x": x, "name": "polygamma_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Polygamma"] = tf_raw_ops_polygamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Polygamma'.")

check_valid('tf.raw_ops.Polygamma', generated_inputs['tf.raw_ops.Polygamma'], lib="tf", suffix=0)
