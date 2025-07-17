
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dynamic_partition_inputs():
    list_of_inputs = []

    # Input 1: Scalar partitions
    data = np.array([10, 20], dtype=np.int32)
    partitions = np.array(1, dtype=np.int32)
    num_partitions = 2
    name = "scalar_partition"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector partitions
    data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    partitions = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    num_partitions = 2
    name = "vector_partition"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional data and partitions
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    partitions = np.array([0, 1, 0], dtype=np.int32)
    num_partitions = 2
    name = "multi_dim_data"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher num_partitions
    data = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    partitions = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    num_partitions = 3
    name = "higher_num_partitions"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger data and partitions
    data = np.arange(24, dtype=np.int32).reshape((4, 3, 2))
    partitions = np.array([0, 1, 0, 2], dtype=np.int32)
    num_partitions = 3
    name = "larger_data"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float data type
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    partitions = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    num_partitions = 2
    name = "float_data"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More partitions than data elements
    data = np.array([1, 2, 3], dtype=np.int32)
    partitions = np.array([0, 1, 2], dtype=np.int32)
    num_partitions = 4
    name = "more_partitions"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Remove string data due to dtype issues
    # data = np.array(["a", "b", "c", "d", "e"], dtype=np.str_)
    # partitions = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    # num_partitions = 2
    # name = "string_data"
    # input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D partitions
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    partitions = np.array([[0, 1], [1, 0]], dtype=np.int32)
    num_partitions = 2
    name = "2d_partitions"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shape between data and partitions (but data.shape starts with partitions.shape)
    data = np.arange(12, dtype=np.int32).reshape((2, 3, 2))
    partitions = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
    num_partitions = 2
    name = "shape_mismatch"
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
