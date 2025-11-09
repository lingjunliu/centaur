
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_bitwise_bitwise_xor_inputs():
    list_of_inputs = []

    x = np.array([0, 5, -3, 14], dtype=np.int32)
    y = np.array([5, 0, 7, 11], dtype=np.int32)
    name = "xor_int32_vec"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2, -3], [4, 5, -6]], dtype=np.int64)
    y = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64)
    name = "xor_int64_mat"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array(5, dtype=np.int32)
    name = "xor_int32_broadcast_scalar"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.int64)  # shape (2,1,3)
    y = np.array([[[1], [2]]], dtype=np.int64)                # shape (1,2,1)
    name = "xor_int64_3d_broadcast"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-1, dtype=np.int32)
    y = np.array([[0, 1], [2, 3]], dtype=np.int32)
    name = "xor_int32_scalar_mat"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, -(2**30), 2**30 - 1], dtype=np.int32)
    y = np.array([2**29, -(2**29), 0], dtype=np.int32)
    name = "xor_int32_large_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.zeros((3, 4), dtype=np.int64)
    y = np.ones((3, 4), dtype=np.int64)
    name = "xor_int64_zeros_ones"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1], [2], [3]], dtype=np.int32)   # shape (3,1)
    y = np.array([[4, 5, 6, 7]], dtype=np.int32)    # shape (1,4)
    name = "xor_int32_broadcast_2d"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[1, 2]]], [[[3, 4]]]], dtype=np.int64)  # shape (2,1,1,2)
    y = np.array([[[[5, 6]]]], dtype=np.int64)              # shape (1,1,1,2)
    name = "xor_int64_4d_broadcast"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[6, 5, 4], [-3, -2, -1]], dtype=np.int32)
    name = "xor_int32_neg_pos"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(7, dtype=np.int64)
    y = np.array(13, dtype=np.int64)
    name = "xor_int64_scalar_scalar"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0, -1, 2**40 - 1]], dtype=np.int64)
    y = np.array([[2**35, 2**35 - 1, 0]], dtype=np.int64)
    name = "xor_int64_large_mixed"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.bitwise.bitwise_xor"] = tf_bitwise_bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.bitwise.bitwise_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_xor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.bitwise.bitwise_xor', generated_inputs['tf.bitwise.bitwise_xor'], lib="tf", suffix=0)
