
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_concat_inputs():
    list_of_inputs = []

    # Input 1: Concatenate along axis 0 (rows)
    t1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
    values = [t1, t2]
    axis = 0
    name = "concat_example_1"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Concatenate along axis 1 (columns)
    t1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
    values = [t1, t2]
    axis = 1
    name = "concat_example_2"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensors, concatenate along axis 0
    t1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    t2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    values = [t1, t2]
    axis = 0
    name = "concat_example_3"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensors, concatenate along axis 1
    t1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    t2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    values = [t1, t2]
    axis = 1
    name = "concat_example_4"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensors, concatenate along axis 2
    t1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    t2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    values = [t1, t2]
    axis = 2
    name = "concat_example_5"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative axis (-1)
    t1 = np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]], dtype=np.int32)
    t2 = np.array([[[7, 4], [8, 4]], [[2, 10], [15, 11]]], dtype=np.int32)
    values = [t1, t2]
    axis = -1
    name = "concat_example_6"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative axis (-2)
    t1 = np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]], dtype=np.int32)
    t2 = np.array([[[7, 4], [8, 4]], [[2, 10], [15, 11]]], dtype=np.int32)
    values = [t1, t2]
    axis = -2
    name = "concat_example_7"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data types
    t1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    values = [t1, t2]
    axis = 0
    name = "concat_example_8"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single Tensor
    t1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    values = [t1]
    axis = 1
    name = "concat_example_9"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single tensor with different shape and axis
    t1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32).reshape((2, 3))
    values = [t1]
    axis = 1
    name = "concat_example_10"
    input_dict = {"values": values, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.concat"] = tf_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.concat'.")

check_valid('tf.concat', generated_inputs['tf.concat'], lib="tf", suffix=0)
