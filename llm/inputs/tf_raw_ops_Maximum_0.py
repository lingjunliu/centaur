
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_maximum_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic int32
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 1, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values, float64
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float64)
    y = np.array([2.0, -1.0, 4.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero values, int64
    x = np.array([0, 2, 0], dtype=np.int64)
    y = np.array([2, 0, 4], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays, uint8
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[2, 1], [4, 3]], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes (broadcasting), float32
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays, int16
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    y = np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]], dtype=np.int16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16) # Simulate bfloat16 with float16 as bfloat16 is not directly supported by numpy
    y = np.array([2.0, 1.0, 4.0], dtype=np.float16) # Simulate bfloat16 with float16 as bfloat16 is not directly supported by numpy
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32
    x = np.array([1, 2, 3], dtype=np.uint32)
    y = np.array([2, 1, 0], dtype=np.uint32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: name, uint64
    x = np.array([1, 2], dtype=np.uint64)
    y = np.array([3, 1], dtype=np.uint64)
    input_dict = {"x": x, "y": y, "name": "my_maximum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Maximum"] = tf_raw_ops_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Maximum'.")

check_valid('tf.raw_ops.Maximum', generated_inputs['tf.raw_ops.Maximum'], lib="tf", suffix=0)
