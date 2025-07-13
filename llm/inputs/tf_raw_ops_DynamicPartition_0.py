
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dynamic_partition_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    partitions = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    num_partitions = 2
    name = "partition1"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    partitions = np.array([0, 1, 0], dtype=np.int32)
    num_partitions = 2
    name = "partition2"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    partitions = np.array([0, 1], dtype=np.int32)
    num_partitions = 3
    name = "partition3"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    partitions = np.array([1, 0, 1], dtype=np.int32)
    num_partitions = 3
    name = "partition4"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([10, 20], dtype=np.int32)
    partitions = np.array(1, dtype=np.int32)
    num_partitions = 2
    name = "partition5"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    partitions = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32)
    num_partitions = 4
    name = "partition6"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],[[9,10],[11,12]]], dtype=np.int32)
    partitions = np.array([0, 1, 0], dtype=np.int32)
    num_partitions = 2
    name = "partition7"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([10, 20, 30], dtype=np.int32)
    partitions = np.array([0, 0, 0], dtype=np.int32)
    num_partitions = 1
    name = "partition8"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    partitions = np.array([0, 1], dtype=np.int32)
    num_partitions = 5
    name = "partition9"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    partitions = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    num_partitions = 2
    name = "partition10"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DynamicPartition"] = tf_raw_ops_dynamic_partition_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DynamicPartition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DynamicPartition'.")

check_valid('tf.raw_ops.DynamicPartition', generated_inputs['tf.raw_ops.DynamicPartition'], lib="tf", suffix=0)
