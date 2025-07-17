
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_placeholder_inputs():
    list_of_inputs = []

    # Input 1
    dtype = tf.float32
    shape = None
    name = "placeholder_1"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtype = tf.int32
    shape = [2, 3]
    name = "placeholder_2"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtype = tf.bool
    shape = [1, 5, 7]
    name = "placeholder_3"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtype = tf.string
    shape = [4]
    name = "placeholder_4"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtype = tf.complex64
    shape = [2, 2, 2, 2]
    name = "placeholder_5"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtype = tf.int64
    shape = []  # Scalar shape
    name = "placeholder_6"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtype = tf.uint8
    shape = [100]
    name = "placeholder_7"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtype = tf.float64
    shape = [1, 1, 1, 1, 1]
    name = "placeholder_8"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtype = tf.qint8
    shape = [3, 5]
    name = "placeholder_9"
    input_dict = {"dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtype = tf.resource
    shape = None
    name = "placeholder_10"
    input_dict = {"dtype": dtype, "shape": None, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    dtype = tf.string
    shape = None
    name = "placeholder_11"
    input_dict = {"dtype": dtype, "shape": None, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    dtype = tf.float16
    shape = []
    name = "placeholder_12"
    input_dict = {"dtype": dtype, "shape": [], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    dtype = tf.variant
    shape = [4,5]
    name = "placeholder_13"
    input_dict = {"dtype": dtype, "shape": [4,5], "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Placeholder"] = tf_raw_ops_placeholder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Placeholder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Placeholder'.")

check_valid('tf.raw_ops.Placeholder', generated_inputs['tf.raw_ops.Placeholder'], lib="tf", suffix=0)
