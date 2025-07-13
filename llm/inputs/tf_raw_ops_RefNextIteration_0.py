
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ref_next_iteration_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 2, 3], dtype=np.int32)
    name = "next_iteration_1"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "next_iteration_2"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([True, False, True], dtype=np.bool_)
    name = "next_iteration_3"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([1, -2, 3, -4], dtype=np.int64)
    name = "next_iteration_4"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    name = "next_iteration_5"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([["a", "b"], ["c", "d"]], dtype=np.string_)
    name = "next_iteration_6"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    name = "next_iteration_7"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([], dtype=np.float32)
    name = "next_iteration_8"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    name = "next_iteration_9"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    data = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    name = "next_iteration_10"
    input_dict = {"data": data, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.RefNextIteration"] = tf_raw_ops_ref_next_iteration_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.RefNextIteration' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefNextIteration'.")

check_valid('tf.raw_ops.RefNextIteration', generated_inputs['tf.raw_ops.RefNextIteration'], lib="tf", suffix=0)
