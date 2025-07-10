
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_enumerate_dataset_inputs():
    list_of_inputs = []

    # Input 1: start = 0
    start = tf.constant(0, dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: start = 1
    start = tf.constant(1, dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: start = 10
    start = tf.constant(10, dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: start = -1
    start = tf.constant(-1, dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: start = -10
    start = tf.constant(-10, dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: start as numpy
    start = tf.constant(np.int64(5), dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: start as numpy negative
    start = tf.constant(np.int64(-5), dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: start with rank 0 tensor
    start = tf.constant(100, dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: start with large value
    start = tf.constant(np.iinfo(np.int64).max // 2, dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: start with small value
    start = tf.constant(np.iinfo(np.int64).min // 2, dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: start = numpy.int64(0)
    start = tf.constant(np.int64(0), dtype=tf.int64)
    input_dict = {'start': start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.enumerate_dataset"] = tf_data_experimental_enumerate_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.enumerate_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.enumerate_dataset'.")

check_valid('tf.data.experimental.enumerate_dataset', generated_inputs['tf.data.experimental.enumerate_dataset'], lib="tf", suffix=0)
