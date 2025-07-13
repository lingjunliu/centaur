
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_parallel_dynamic_stitch_inputs():
    list_of_inputs = []

    # Input 1
    indices = [np.array([0, 2, 1], dtype=np.int32)]
    data = [np.array([10, 20, 30], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices = [np.array([[0, 1], [2, 3]], dtype=np.int32)]
    data = [np.array([[1, 2], [3, 4]], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([100], dtype=np.int32), np.array([200], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices = [np.array([2, 0], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([5, 7], dtype=np.int32), np.array([9], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
    data = [np.array([1, 2], dtype=np.float32), np.array([3, 4], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices = [np.array([[0, 1], [2, 3]], dtype=np.int32), np.array([[4, 5], [6, 7]], dtype=np.int32)]
    data = [np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices = [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    data = [np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices = [np.array([0], dtype=np.int32), np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)]
    data = [np.array([10], dtype=np.int32), np.array([20, 30], dtype=np.int32), np.array([40], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices = [np.array([0, 1, 2], dtype=np.int32)]
    data = [np.array([1, 2, 3], dtype=np.int64)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([10], dtype=np.float64), np.array([20], dtype=np.float64)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    indices = [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)]
    data = [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    indices = [np.array([0, 2], dtype=np.int32), np.array([1], dtype=np.int32)]
    data = [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13 - Adding a different shape and data type
    indices = [np.array([0, 1], dtype=np.int32), np.array([2,3], dtype=np.int32)]
    data = [np.array([1.1, 2.2], dtype=np.float32), np.array([3.3, 4.4], dtype=np.float32)]
    input_dict = {"indices": indices, "data": data, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ParallelDynamicStitch"] = tf_raw_ops_parallel_dynamic_stitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ParallelDynamicStitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParallelDynamicStitch'.")

check_valid('tf.raw_ops.ParallelDynamicStitch', generated_inputs['tf.raw_ops.ParallelDynamicStitch'], lib="tf", suffix=0)
