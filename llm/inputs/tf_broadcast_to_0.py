
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_to_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant([1, 2, 3])
    shape_tensor = tf.constant([2, 3])
    name = "broadcast_example_1"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant([[1, 2], [3, 4]])
    shape_tensor = tf.constant([2, 2, 2])
    name = "broadcast_example_2"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(1)
    shape_tensor = tf.constant([2, 3, 4])
    name = "broadcast_example_3"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant([[[1]]])
    shape_tensor = tf.constant([2, 3, 1, 1])
    name = "broadcast_example_4"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = tf.constant([1, 2, 3], dtype=tf.float32)
    shape_tensor = tf.constant([2, 3])
    name = "broadcast_example_5"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = tf.constant([[1], [2]])
    shape_tensor = tf.constant([2, 3])
    name = "broadcast_example_6"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = tf.constant([[[1, 2]]])
    shape_tensor = tf.constant([2, 1, 1, 2])
    name = "broadcast_example_7"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = tf.constant([1, 2, 3, 4])
    shape_tensor = tf.constant([5, 4])
    name = "broadcast_example_8"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = tf.constant([[1, 2, 3]])
    shape_tensor = tf.constant([2, 1, 3])
    name = "broadcast_example_9"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = tf.constant([[[1], [2]]])
    shape_tensor = tf.constant([2, 1, 2, 3])
    name = "broadcast_example_10"
    input_dict = {"input": input_tensor.numpy(), "shape": shape_tensor.numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.broadcast_to"] = tf_broadcast_to_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.broadcast_to' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.broadcast_to'.")

check_valid('tf.broadcast_to', generated_inputs['tf.broadcast_to'], lib="tf", suffix=0)
