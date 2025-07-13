
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_inplace_add_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[5, 6]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple indices
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    i = np.array([0, 2], dtype=np.int32)
    v = np.array([[5, 6], [7, 8]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type (int32)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[5, 6]], dtype=np.int32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimensions
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[[9, 10], [11, 12]]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: Negative values
    x = np.array([[1, -2], [-3, 4]], dtype=np.float32)
    i = np.array([1], dtype=np.int32)
    v = np.array([[-5, 6]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger x
    x = np.random.rand(10, 5).astype(np.float32)
    i = np.array([2, 5, 7], dtype=np.int32)
    v = np.random.rand(3, 5).astype(np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: i close to boundary of x
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    i = np.array([2], dtype=np.int32)
    v = np.array([[7, 8]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data type (float64)
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    i = np.array([0], dtype=np.int32)
    v = np.array([[5, 6]], dtype=np.float64)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: name
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    i = np.array([0], dtype=np.int32)
    v = np.array([[5, 6]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": "my_inplace_add"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More rows
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    i = np.array([0, 2], dtype=np.int32)
    v = np.array([[9, 10], [11, 12]], dtype=np.float32)
    input_dict = {"x": x, "i": i, "v": v, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.InplaceAdd"] = tf_raw_ops_inplace_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.InplaceAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InplaceAdd'.")

check_valid('tf.raw_ops.InplaceAdd', generated_inputs['tf.raw_ops.InplaceAdd'], lib="tf", suffix=0)
