
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ensure_shape_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 2, 3, 4, 5])
    shape = [5]
    name = "ensure_shape_op_1"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 2
    input_tensor = np.array([[1, 2], [3, 4]])
    shape = [2, 2]
    name = "ensure_shape_op_2"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 3
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    shape = [2, 2, 2]
    name = "ensure_shape_op_3"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 4
    input_tensor = np.array([1.0, 2.0, 3.0])
    shape = [3]
    name = "ensure_shape_op_4"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 5
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    shape = [2, 2]
    name = "ensure_shape_op_5"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 6
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    shape = [5]
    name = "ensure_shape_op_6"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 7
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    shape = [5]
    name = "ensure_shape_op_7"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 8
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    shape = [2, 2, 2]
    name = "ensure_shape_op_8"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 9
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape = [2, 3]
    name = "ensure_shape_op_9"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    # Input 10
    input_tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32).reshape((2, 2, 2))
    shape = [2, 2, 2]
    name = "ensure_shape_op_10"
    input_dict = {"input": input_tensor, "shape": shape, "name": name}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_ensure_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EnsureShape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EnsureShape'.")

check_valid('tf.raw_ops.EnsureShape', generated_inputs['tf.raw_ops.EnsureShape'], lib="tf", suffix=0)
