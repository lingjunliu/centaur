
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_conditional_accumulator_inputs():
    list_of_inputs = []

    # Input 1
    dtype = tf.float32
    shape = []
    container = ""
    shared_name = ""
    reduction_type = "MEAN"
    name = ""
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtype = tf.int32
    shape = [2, 3]
    container = "test_container"
    shared_name = "test_shared_name"
    reduction_type = "SUM"
    name = "accumulator_1"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtype = tf.float64
    shape = [1, 2, 3]
    container = ""
    shared_name = "shared_name_2"
    reduction_type = "MEAN"
    name = "accumulator_2"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtype = tf.complex64
    shape = [5]
    container = "container_3"
    shared_name = ""
    reduction_type = "SUM"
    name = "accumulator_3"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtype = tf.int64
    shape = [2, 2, 2, 2]
    container = "container_4"
    shared_name = "shared_name_4"
    reduction_type = "MEAN"
    name = "accumulator_4"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtype = tf.uint8
    shape = [10]
    container = ""
    shared_name = ""
    reduction_type = "SUM"
    name = "accumulator_5"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    dtype = tf.bfloat16
    shape = [1]
    container = "container_6"
    shared_name = "shared_name_6"
    reduction_type = "MEAN"
    name = "accumulator_6"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    dtype = tf.half
    shape = [4,5,6,7,8]
    container = ""
    shared_name = ""
    reduction_type = "SUM"
    name = "accumulator_7"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtype = tf.qint32
    shape = [1, 1]
    container = "container_8"
    shared_name = "shared_name_8"
    reduction_type = "MEAN"
    name = "accumulator_8"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtype = tf.quint8
    shape = []
    container = ""
    shared_name = ""
    reduction_type = "SUM"
    name = "accumulator_9"
    input_dict = {"dtype": dtype, "shape": shape, "container": container, "shared_name": shared_name, "reduction_type": reduction_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ConditionalAccumulator"] = tf_raw_ops_conditional_accumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ConditionalAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ConditionalAccumulator'.")

check_valid('tf.raw_ops.ConditionalAccumulator', generated_inputs['tf.raw_ops.ConditionalAccumulator'], lib="tf", suffix=0)
