
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_RefSelect_inputs():
    list_of_inputs = []

    # Input 1
    index = np.array(0, dtype=np.int32)
    inputs = [tf.Variable(np.array([1, 2, 3], dtype=np.float32)), tf.Variable(np.array([4, 5, 6], dtype=np.float32))]
    input_dict = {"index": index, "inputs": inputs, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    index = np.array(1, dtype=np.int32)
    inputs = [tf.Variable(np.array([1, 2, 3], dtype=np.int32)), tf.Variable(np.array([4, 5, 6], dtype=np.int32))]
    input_dict = {"index": index, "inputs": inputs, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    index = np.array(0, dtype=np.int32)
    inputs = [tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float64)), tf.Variable(np.array([[5, 6], [7, 8]], dtype=np.float64))]
    input_dict = {"index": index, "inputs": inputs, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    index = np.array(1, dtype=np.int32)
    inputs = [tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int64)), tf.Variable(np.array([[5, 6], [7, 8]], dtype=np.int64))]
    input_dict = {"index": index, "inputs": inputs, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    index = np.array(0, dtype=np.int32)
    inputs = [tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)), tf.Variable(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32))]
    input_dict = {"index": index, "inputs": inputs, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    index = np.array(1, dtype=np.int32)
    inputs = [tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)), tf.Variable(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32))]
    input_dict = {"index": index, "inputs": inputs, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    index = np.array(0, dtype=np.int32)
    inputs = [tf.Variable(np.array([1], dtype=np.float32)), tf.Variable(np.array([2], dtype=np.float32))]
    input_dict = {"index": index, "inputs": inputs, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    index = np.array(1, dtype=np.int32)
    inputs = [tf.Variable(np.array([1], dtype=np.int32)), tf.Variable(np.array([2], dtype=np.int32))]
    input_dict = {"index": index, "inputs": inputs, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More tensors
    index = np.array(0, dtype=np.int32)
    inputs = [tf.Variable(np.array([1, 2], dtype=np.float32)), tf.Variable(np.array([3, 4], dtype=np.float32)), tf.Variable(np.array([5, 6], dtype=np.float32))]
    input_dict = {"index": index, "inputs": inputs, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different index
    index = np.array(1, dtype=np.int32)
    inputs = [tf.Variable(np.array([1, 2], dtype=np.int32)), tf.Variable(np.array([3, 4], dtype=np.int32)), tf.Variable(np.array([5, 6], dtype=np.int32))]
    input_dict = {"index": index, "inputs": inputs, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_RefSelect_inputs()
generated_inputs["tf.raw_ops.RefSelect"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RefSelect' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSelect'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RefSelect', generated_inputs['tf.raw_ops.RefSelect'], lib="tf", suffix=0)
