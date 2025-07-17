
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_merge_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two float tensors
    inputs = [tf.constant([1.0, 2.0, 3.0], dtype=tf.float32), tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)]
    input_dict = {"inputs": inputs, "name": "merge_example_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two int32 tensors
    inputs = [tf.constant([1, 2, 3], dtype=tf.int32), tf.constant([4, 5, 6], dtype=tf.int32)]
    input_dict = {"inputs": inputs, "name": "merge_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two bool tensors
    inputs = [tf.constant([True, False], dtype=tf.bool), tf.constant([False, True], dtype=tf.bool)]
    input_dict = {"inputs": inputs, "name": "merge_example_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two string tensors
    inputs = [tf.constant(["a", "b"]), tf.constant(["c", "d"])]
    input_dict = {"inputs": inputs, "name": "merge_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More than two tensors (3 tensors of type int64)
    inputs = [tf.constant([1, 2], dtype=tf.int64), tf.constant([3, 4], dtype=tf.int64), tf.constant([5, 6], dtype=tf.int64)]
    input_dict = {"inputs": inputs, "name": "merge_example_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Two float64 tensors
    inputs = [tf.constant([1.0, 2.0], dtype=tf.float64), tf.constant([3.0, 4.0], dtype=tf.float64)]
    input_dict = {"inputs": inputs, "name": "merge_example_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Two tensors with negative values.
    inputs = [tf.constant([-1, -2], dtype=tf.int32), tf.constant([-3, -4], dtype=tf.int32)]
    input_dict = {"inputs": inputs, "name": "merge_example_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Two tensors, shape (2, 2)
    inputs = [tf.constant([[1, 2], [3, 4]], dtype=tf.int32), tf.constant([[5, 6], [7, 8]], dtype=tf.int32)]
    input_dict = {"inputs": inputs, "name": "merge_example_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Two tensors of different shapes, but same type
    inputs = [tf.constant(1, dtype=tf.int32), tf.constant([2,3], dtype=tf.int32)]
    input_dict = {"inputs": inputs, "name": "merge_example_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Merge"] = tf_raw_ops_merge_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Merge' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Merge'.")

check_valid('tf.raw_ops.Merge', generated_inputs['tf.raw_ops.Merge'], lib="tf", suffix=0)
