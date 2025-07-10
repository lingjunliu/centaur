
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_prod_inputs():
    list_of_inputs = []

    # Input 1
    data = tf.constant([1, 2, 3, 4], dtype=tf.int32).numpy()
    segment_ids = tf.constant([0, 0, 1, 1], dtype=tf.int32).numpy()
    num_segments = tf.constant(2, dtype=tf.int32).numpy()
    name = "example_1"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=tf.int32).numpy()
    segment_ids = tf.constant([0, 1, 0], dtype=tf.int32).numpy()
    num_segments = tf.constant(2, dtype=tf.int32).numpy()
    name = "example_2"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32).numpy()
    segment_ids = tf.constant([0, 0, 1, 1, 2], dtype=tf.int32).numpy()
    num_segments = tf.constant(3, dtype=tf.int32).numpy()
    name = "example_3"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = tf.constant([1, 2, 3, 4], dtype=tf.int64).numpy()
    segment_ids = tf.constant([0, 0, 1, 0], dtype=tf.int32).numpy()
    num_segments = tf.constant(2, dtype=tf.int32).numpy()
    name = "example_4"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int32).numpy()
    segment_ids = tf.constant([0, 1, 0], dtype=tf.int32).numpy()
    num_segments = tf.constant(3, dtype=tf.int32).numpy()
    name = "example_5"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data type (float64)
    data = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float64).numpy()
    segment_ids = tf.constant([0, 0, 1, 1], dtype=tf.int32).numpy()
    num_segments = tf.constant(2, dtype=tf.int32).numpy()
    name = "example_6"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative segment IDs (ignored)
    data = tf.constant([1, 2, 3, 4], dtype=tf.int32).numpy()
    segment_ids = tf.constant([-1, 0, 1, -1], dtype=tf.int32).numpy()
    num_segments = tf.constant(2, dtype=tf.int32).numpy()
    name = "example_7"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 data type
    data = tf.constant([1+1j, 2+2j, 3+3j, 4+4j], dtype=tf.complex64).numpy()
    segment_ids = tf.constant([0, 0, 1, 1], dtype=tf.int32).numpy()
    num_segments = tf.constant(2, dtype=tf.int32).numpy()
    name = "example_8"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9: uint8 data type
    data = tf.constant([1, 2, 3, 4], dtype=tf.uint8).numpy()
    segment_ids = tf.constant([0, 0, 1, 1], dtype=tf.int32).numpy()
    num_segments = tf.constant(2, dtype=tf.int32).numpy()
    name = "example_9"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: int16 data type
    data = tf.constant([1, 2, 3, 4], dtype=tf.int16).numpy()
    segment_ids = tf.constant([0, 0, 1, 1], dtype=tf.int32).numpy()
    num_segments = tf.constant(2, dtype=tf.int32).numpy()
    name = "example_10"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.unsorted_segment_prod"] = tf_math_unsorted_segment_prod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.unsorted_segment_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_prod'.")

check_valid('tf.math.unsorted_segment_prod', generated_inputs['tf.math.unsorted_segment_prod'], lib="tf", suffix=0)
