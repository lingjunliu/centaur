
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_unique_with_counts_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    x = np.array([1, 1, 2, 4, 4, 4, 7, 8, 8], dtype=np.int32)
    out_idx = tf.int32
    name = "unique_counts_1"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (float)
    x = np.array([1.0, 1.0, 2.0, 4.0, 4.0, 4.0, 7.0, 8.0, 8.0], dtype=np.float32)
    out_idx = tf.int32
    name = "unique_counts_2"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different out_idx (int64)
    x = np.array([1, 1, 2, 4, 4, 4, 7, 8, 8], dtype=np.int32)
    out_idx = tf.int64
    name = "unique_counts_3"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With negative values
    x = np.array([-1, -1, 0, 1, 1, 2, -2, -2, -2], dtype=np.int32)
    out_idx = tf.int32
    name = "unique_counts_4"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All same values
    x = np.array([5, 5, 5, 5, 5], dtype=np.int32)
    out_idx = tf.int32
    name = "unique_counts_5"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Already unique values
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    out_idx = tf.int32
    name = "unique_counts_6"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Empty array
    x = np.array([], dtype=np.int32)
    out_idx = tf.int32
    name = "unique_counts_7"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64
    x = np.array([1.1, 1.1, 2.2, 3.3, 3.3, 4.4], dtype=np.float64)
    out_idx = tf.int32
    name = "unique_counts_8"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large numbers
    x = np.array([1000000000, 1000000000, 2000000000, 3000000000], dtype=np.int64)
    out_idx = tf.int32
    name = "unique_counts_9"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mix of positive and negative floats
    x = np.array([-1.5, -1.5, 0.0, 1.5, 1.5, 2.5, -2.5, -2.5, -2.5], dtype=np.float32)
    out_idx = tf.int32
    name = "unique_counts_10"
    input_dict = {"x": x, "out_idx": out_idx, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.unique_with_counts"] = tf_unique_with_counts_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.unique_with_counts' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.unique_with_counts'.")

check_valid('tf.unique_with_counts', generated_inputs['tf.unique_with_counts'], lib="tf", suffix=0)
