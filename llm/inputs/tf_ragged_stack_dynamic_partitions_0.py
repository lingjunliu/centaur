
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_stack_dynamic_partitions_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array(['a', 'b', 'c', 'd', 'e'], dtype=object)
    partitions = np.array([3, 0, 2, 2, 3], dtype=np.int32)
    num_partitions = 5
    name = "test_1"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([1, 2, 3, 4, 5])
    partitions = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    num_partitions = 3
    name = "test_2"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multidimensional data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    partitions = np.array([0, 1, 0, 1], dtype=np.int32)
    num_partitions = 2
    name = "test_3"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type
    data = np.array([1.0, 2.0, 3.0, 4.0])
    partitions = np.array([0, 1, 0, 1], dtype=np.int32)
    num_partitions = 2
    name = "test_4"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty partitions
    data = np.array([1, 2, 3, 4])
    partitions = np.array([1, 1, 1, 1], dtype=np.int32)
    num_partitions = 3
    name = "test_5"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single partition
    data = np.array([1, 2, 3])
    partitions = np.array([0, 0, 0], dtype=np.int32)
    num_partitions = 1
    name = "test_6"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different partition numbers
    data = np.array([1, 2, 3, 4, 5])
    partitions = np.array([2, 0, 1, 2, 0], dtype=np.int32)
    num_partitions = 4
    name = "test_7"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Multidimensional data with more partitions
    data = np.array([[1, 2], [3, 4], [5, 6]])
    partitions = np.array([0, 2, 1], dtype=np.int32)
    num_partitions = 3
    name = "test_8"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty data
    data = np.array([])
    partitions = np.array([], dtype=np.int32)
    num_partitions = 3
    name = "test_9"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger num_partitions
    data = np.array([1, 2, 3])
    partitions = np.array([0, 0, 0], dtype=np.int32)
    num_partitions = 5
    name = "test_10"
    input_dict = {"data": data, "partitions": partitions, "num_partitions": num_partitions, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.stack_dynamic_partitions"] = tf_ragged_stack_dynamic_partitions_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.stack_dynamic_partitions' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.stack_dynamic_partitions'.")

check_valid('tf.ragged.stack_dynamic_partitions', generated_inputs['tf.ragged.stack_dynamic_partitions'], lib="tf", suffix=0)
