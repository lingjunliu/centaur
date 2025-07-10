
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_size_inputs():
    list_of_inputs = []

    # Input 1: Basic tensor
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    out_type = tf.int32
    name = "size_example_1"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multi-dimensional tensor
    input_tensor = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    out_type = tf.int64
    name = "size_example_2"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    out_type = tf.int32
    name = "size_example_3"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with different data type (float)
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out_type = tf.int32
    name = "size_example_4"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with different data type (string) - Removed due to dtype issues
    # input_tensor = np.array(["a", "b", "c"])
    # out_type = tf.int32
    # name = "size_example_5"
    # input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor
    input_tensor = np.array([], dtype=np.int32)
    out_type = tf.int32
    name = "size_example_6"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with specified int64 out_type
    input_tensor = np.array([1, 2, 3], dtype=np.int64)
    out_type = tf.int64
    name = "size_example_7"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor
    input_tensor = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], dtype=np.int32)
    out_type = tf.int32
    name = "size_example_8"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with negative values
    input_tensor = np.array([-1, -2, 3, -4, 5], dtype=np.int32)
    out_type = tf.int32
    name = "size_example_9"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with only one element
    input_tensor = np.array([5], dtype=np.int32)
    out_type = tf.int32
    name = "size_example_10"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.size"] = tf_size_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.size'.")

check_valid('tf.size', generated_inputs['tf.size'], lib="tf", suffix=0)
