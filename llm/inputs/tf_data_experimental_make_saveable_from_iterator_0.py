
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_make_saveable_from_iterator_inputs():
    list_of_inputs = []

    # Input 1
    init_tensor = tf.constant(1)
    input_dict = {"iterator": init_tensor, "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    init_tensor = tf.constant(2)
    input_dict = {"iterator": init_tensor, "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    init_tensor = tf.constant(3)
    input_dict = {"iterator": init_tensor, "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    init_tensor = tf.constant(4)
    input_dict = {"iterator": init_tensor, "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    init_tensor = tf.constant(5)
    input_dict = {"iterator": init_tensor, "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    init_tensor = tf.constant(6)
    input_dict = {"iterator": init_tensor, "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    init_tensor = tf.constant(7)
    input_dict = {"iterator": init_tensor, "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    init_tensor = tf.constant(8)
    input_dict = {"iterator": init_tensor, "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    init_tensor = tf.constant(9)
    input_dict = {"iterator": init_tensor, "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    init_tensor = tf.constant(10)
    input_dict = {"iterator": init_tensor, "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    init_tensor = tf.constant(11)
    input_dict = {"iterator": init_tensor, "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    init_tensor = tf.constant([[1, 2], [3, 4]])
    input_dict = {"iterator": init_tensor, "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    init_tensor = tf.constant([1.0, 2.0, 3.0])
    input_dict = {"iterator": init_tensor, "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14
    init_tensor = tf.constant(-1)
    input_dict = {"iterator": init_tensor, "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.make_saveable_from_iterator"] = tf_data_experimental_make_saveable_from_iterator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.make_saveable_from_iterator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.make_saveable_from_iterator'.")

check_valid('tf.data.experimental.make_saveable_from_iterator', generated_inputs['tf.data.experimental.make_saveable_from_iterator'], lib="tf", suffix=0)
