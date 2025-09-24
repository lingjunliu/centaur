
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_next_after_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensors
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.1, 2.1, 3.1], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64 tensors
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2 = np.array([1.1, 2.1, 3.1], dtype=np.float64)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x2 = np.array([-1.1, -2.1, -3.1], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero values
    x1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional tensors
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Same values for x1 and x2
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different magnitudes
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small differences
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.000001, 2.000001, 3.000001], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Using name
    x1 = np.array([1.0], dtype=np.float32)
    x2 = np.array([1.1], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "name": "my_next_after"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger multi-dimensional tensors
    x1 = np.random.rand(3, 4, 5).astype(np.float64)
    x2 = np.random.rand(3, 4, 5).astype(np.float64)
    input_dict = {"x1": x1, "x2": x2, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.NextAfter"] = tf_raw_ops_next_after_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.NextAfter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NextAfter'.")

check_valid('tf.raw_ops.NextAfter', generated_inputs['tf.raw_ops.NextAfter'], lib="tf", suffix=0)
