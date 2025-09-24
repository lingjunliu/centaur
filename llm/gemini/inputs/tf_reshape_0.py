
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_reshape_inputs():
    list_of_inputs = []

    # Input 1
    tensor = tf.constant(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    shape = tf.constant(np.array([6], dtype=np.int32))
    name = "reshape_example_1"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = tf.constant(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))
    shape = tf.constant(np.array([2, 3], dtype=np.int32))
    name = "reshape_example_2"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32))
    shape = tf.constant(np.array([2, 4], dtype=np.int32))
    name = "reshape_example_3"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32))
    shape = tf.constant(np.array([-1, 2], dtype=np.int32))
    name = "reshape_example_4"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32))
    shape = tf.constant(np.array([3, 3], dtype=np.int32))
    name = "reshape_example_5"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = tf.constant(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.int32))
    shape = tf.constant(np.array([-1], dtype=np.int32))
    name = "reshape_example_6"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = tf.constant(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.int32))
    shape = tf.constant(np.array([2, -1], dtype=np.int32))
    name = "reshape_example_7"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = tf.constant(np.array([7], dtype=np.int32))
    shape = tf.constant(np.array([], dtype=np.int32))
    name = "reshape_example_8"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = tf.constant(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))
    shape = tf.constant(np.array([1, 6], dtype=np.int32))
    name = "reshape_example_9"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    tensor = tf.constant(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], dtype=np.int32))
    shape = tf.constant(np.array([2, -1, 3], dtype=np.int32))
    name = "reshape_example_10"
    input_dict = {"tensor": tensor, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.reshape"] = tf_reshape_inputs()
for i in range(len(generated_inputs["tf.reshape"])):
    generated_inputs["tf.reshape"][i]["tensor"] = generated_inputs["tf.reshape"][i]["tensor"].numpy()
    generated_inputs["tf.reshape"][i]["shape"] = generated_inputs["tf.reshape"][i]["shape"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reshape'.")

check_valid('tf.reshape', generated_inputs['tf.reshape'], lib="tf", suffix=0)
