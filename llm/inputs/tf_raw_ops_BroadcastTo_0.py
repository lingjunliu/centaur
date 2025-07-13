
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_broadcast_to_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant([1, 2, 3], dtype=tf.int32).numpy()
    shape_tensor = tf.constant([2, 3], dtype=tf.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant([[1, 2]], dtype=tf.float32).numpy()
    shape_tensor = tf.constant([2, 2], dtype=tf.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(10, dtype=tf.int64).numpy()
    shape_tensor = tf.constant([5], dtype=tf.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant([[[1]]], dtype=tf.float64).numpy()
    shape_tensor = tf.constant([2, 1, 1], dtype=tf.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = tf.constant([1, 2], dtype=tf.int32).numpy()
    shape_tensor = tf.constant([2, 2], dtype=tf.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = tf.constant([[1], [2]], dtype=tf.float32).numpy()
    shape_tensor = tf.constant([2, 3], dtype=tf.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = tf.constant([1], dtype=tf.int64).numpy()
    shape_tensor = tf.constant([1, 5], dtype=tf.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = tf.constant([1,2,3,4], dtype=tf.int32).numpy()
    shape_tensor = tf.constant([2,2,4], dtype=tf.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = tf.constant(np.array([1,2,3]), dtype=tf.float32).numpy()
    shape_tensor = tf.constant(np.array([3,3]), dtype=np.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = tf.constant(np.array([[1],[2]]), dtype=tf.int32).numpy()
    shape_tensor = tf.constant(np.array([2,3]), dtype=np.int32).numpy()
    input_dict = {"input": input_tensor, "shape": shape_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BroadcastTo"] = tf_raw_ops_broadcast_to_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BroadcastTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BroadcastTo'.")

check_valid('tf.raw_ops.BroadcastTo', generated_inputs['tf.raw_ops.BroadcastTo'], lib="tf", suffix=0)
