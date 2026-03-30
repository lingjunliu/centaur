
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RefNextIteration_inputs():
    list_of_inputs = []

    # Input 1
    data = tf.Variable(np.array([1, 2, 3], dtype=np.int32))
    name = "next_iter_1"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    name = "next_iter_2"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = tf.Variable(np.array([True, False, True], dtype=np.bool_))
    name = "next_iter_3"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    name = "next_iter_4"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = tf.Variable(np.array([["a", "b"], ["c", "d"]], dtype=np.unicode_))
    name = "next_iter_5"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = tf.Variable(np.array([-1, -2, -3], dtype=np.int32))
    name = "next_iter_6"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = tf.Variable(np.array([1.5, 2.5, 3.5], dtype=np.float64))
    name = "next_iter_7"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32))
    name = "next_iter_8"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = tf.Variable(np.array([], dtype=np.int32))
    name = "next_iter_9"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = tf.Variable(np.array([1], dtype=np.int32))
    name = "next_iter_10"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RefNextIteration"] = tf_raw_ops_RefNextIteration_inputs()
tf.compat.v1.disable_eager_execution()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RefNextIteration' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefNextIteration'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RefNextIteration', generated_inputs['tf.raw_ops.RefNextIteration'], lib="tf", suffix=0)
