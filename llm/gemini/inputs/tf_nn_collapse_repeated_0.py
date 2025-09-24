
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_collapse_repeated_inputs():
    list_of_inputs = []

    # Input 1
    labels = np.array([[1, 1, 2, 2, 1]]).astype(np.int32)
    seq_length = np.array([5]).astype(np.int32)
    name = "test_collapse_1"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    labels = np.array([[1, 2, 3, 4, 5]]).astype(np.int32)
    seq_length = np.array([5]).astype(np.int32)
    name = "test_collapse_2"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    labels = np.array([[1, 1, 2, 2, 0]]).astype(np.int32)
    seq_length = np.array([4]).astype(np.int32)
    name = "test_collapse_3"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    labels = np.array([[1, 1, 2, 2, 1], [3, 3, 4, 4, 3]]).astype(np.int32)
    seq_length = np.array([5, 5]).astype(np.int32)
    name = "test_collapse_4"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    labels = np.array([[1, 1, 2, 2, 1], [3, 3, 4, 4, 3]]).astype(np.int32)
    seq_length = np.array([3, 4]).astype(np.int32)
    name = "test_collapse_5"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    labels = np.array([[0, 0, 0, 0, 0]]).astype(np.int32)
    seq_length = np.array([5]).astype(np.int32)
    name = "test_collapse_6"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    labels = np.array([[1, 1, 1, 1, 1]]).astype(np.int32)
    seq_length = np.array([5]).astype(np.int32)
    name = "test_collapse_7"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    labels = np.array([[1, 2, 1, 2, 1]]).astype(np.int32)
    seq_length = np.array([5]).astype(np.int32)
    name = "test_collapse_8"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    labels = np.array([[1, 1, 2, 2, 1, 3, 3, 4, 4, 3]]).astype(np.int32)
    seq_length = np.array([10]).astype(np.int32)
    name = "test_collapse_9"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    labels = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]]).astype(np.int32)
    seq_length = np.array([12]).astype(np.int32)
    name = "test_collapse_10"
    input_dict = {"labels": labels, "seq_length": seq_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.collapse_repeated"] = tf_nn_collapse_repeated_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.collapse_repeated' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.collapse_repeated'.")

check_valid('tf.nn.collapse_repeated', generated_inputs['tf.nn.collapse_repeated'], lib="tf", suffix=0)
