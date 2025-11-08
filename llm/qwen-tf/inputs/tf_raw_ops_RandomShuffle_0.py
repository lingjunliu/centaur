
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_random_shuffle_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    value = np.array([[1, 2], [3, 4], [5, 6]])
    input_dict = {
        "value": value,
        "seed": 0,
        "seed2": 0,
        "name": "random_shuffle_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {
        "value": value,
        "seed": 1,
        "seed2": 2,
        "name": "random_shuffle_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    value = np.array([[1, 2], [3, 4], [5, 6]])
    input_dict = {
        "value": value,
        "seed": 3,
        "seed2": 4,
        "name": "random_shuffle_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    value = np.array([1, 2, 3, 4, 5])
    input_dict = {
        "value": value,
        "seed": 5,
        "seed2": 6,
        "name": "random_shuffle_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    input_dict = {
        "value": value,
        "seed": 7,
        "seed2": 8,
        "name": "random_shuffle_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    value = np.array([[1, 2], [3, 4], [5, 6]])
    input_dict = {
        "value": value,
        "seed": 9,
        "seed2": 10,
        "name": "random_shuffle_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    value = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    input_dict = {
        "value": value,
        "seed": 11,
        "seed2": 12,
        "name": "random_shuffle_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    value = np.array([[1, 2], [3, 4], [5, 6]])
    input_dict = {
        "value": value,
        "seed": 13,
        "seed2": 14,
        "name": "random_shuffle_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    input_dict = {
        "value": value,
        "seed": 15,
        "seed2": 16,
        "name": "random_shuffle_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    value = np.array([[1, 2], [3, 4], [5, 6]])
    input_dict = {
        "value": value,
        "seed": 17,
        "seed2": 18,
        "name": "random_shuffle_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomShuffle"] = tf_raw_ops_random_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomShuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffle'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RandomShuffle', generated_inputs['tf.raw_ops.RandomShuffle'], lib="tf", suffix=0)
