
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_enumerate_dataset_inputs():
    list_of_inputs = []

    # Input 1: start = 0
    start = np.int64(0)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: start = 1
    start = np.int64(1)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: start = -1
    start = np.int64(-1)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: start = 100
    start = np.int64(100)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: start = -100
    start = np.int64(-100)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: start = large positive number
    start = np.int64(2**10)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: start = large negative number
    start = np.int64(-(2**10))
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: start = smaller positive
    start = np.int64(10)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: start = smaller negative
    start = np.int64(-10)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: start = a different starting point
    start = np.int64(15)
    input_dict = {"start": tf.convert_to_tensor(start, dtype=tf.int64)}
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
