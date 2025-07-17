
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_enumerate_dataset_inputs():
    list_of_inputs = []

    # Input 1: Scalar start value
    start = np.array(0, dtype=np.int64)
    input_dict = {'start': tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar start value, different starting point
    start = np.array(10, dtype=np.int64)
    input_dict = {'start': tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative start value
    start = np.array(-5, dtype=np.int64)
    input_dict = {'start': tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero start value
    start = np.array(0, dtype=np.int64)
    input_dict = {'start': tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive start value
    start = np.array(10000, dtype=np.int64)
    input_dict = {'start': tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Shaped tensor
    start = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {'start': tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional shaped tensor
    start = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict = {'start': tf.convert_to_tensor(start, dtype=tf.int64)}
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
