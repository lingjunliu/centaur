
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_CountUpTo_inputs():
    list_of_inputs = []

    # Input 1
    ref = tf.Variable(initial_value=0, dtype=tf.int32)
    limit = 5
    name = "count_up_to_1"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = tf.Variable(initial_value=0, dtype=tf.int64)
    limit = 10
    name = "count_up_to_2"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = tf.Variable(initial_value=2, dtype=tf.int32)
    limit = 7
    name = "count_up_to_3"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = tf.Variable(initial_value=5, dtype=tf.int64)
    limit = 12
    name = "count_up_to_4"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = tf.Variable(initial_value=-3, dtype=tf.int32)
    limit = 2
    name = "count_up_to_5"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = tf.Variable(initial_value=-5, dtype=tf.int64)
    limit = 0
    name = "count_up_to_6"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    ref = tf.Variable(initial_value=100, dtype=tf.int32)
    limit = 105
    name = "count_up_to_7"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = tf.Variable(initial_value=1000, dtype=tf.int64)
    limit = 1010
    name = "count_up_to_8"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    ref = tf.Variable(initial_value=2**10, dtype=tf.int32)
    limit = 2**10 + 5
    name = "count_up_to_9"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = tf.Variable(initial_value=2**30, dtype=tf.int64)
    limit = 2**30 + 10
    name = "count_up_to_10"
    input_dict = {"ref": ref, "limit": limit, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.CountUpTo"] = tf_raw_ops_CountUpTo_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.CountUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CountUpTo'.")

check_valid('tf.raw_ops.CountUpTo', generated_inputs['tf.raw_ops.CountUpTo'], lib="tf", suffix=0)
