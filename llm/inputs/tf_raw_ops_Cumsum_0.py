
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cumsum_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, exclusive=False, reverse=False
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = np.int32(0)
    exclusive = False
    reverse = False
    name = "cumsum_1"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, exclusive=True, reverse=False
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    axis = np.int32(0)
    exclusive = True
    reverse = False
    name = "cumsum_2"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, exclusive=False, reverse=True
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    axis = np.int32(0)
    exclusive = False
    reverse = True
    name = "cumsum_3"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, exclusive=True, reverse=True
    x = np.array([1, 2, 3, 4, 5], dtype=np.complex64)
    axis = np.int32(0)
    exclusive = True
    reverse = True
    name = "cumsum_4"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, axis=0
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    axis = np.int32(0)
    exclusive = False
    reverse = False
    name = "cumsum_5"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, axis=1
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    axis = np.int32(1)
    exclusive = False
    reverse = False
    name = "cumsum_6"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, axis=0, exclusive=True, reverse=True
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    axis = np.int32(0)
    exclusive = True
    reverse = True
    name = "cumsum_7"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis=1
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    axis = np.int32(1)
    exclusive = False
    reverse = False
    name = "cumsum_8"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values
    x = np.array([-1, 2, -3, 4, -5], dtype=np.int32)
    axis = np.int32(0)
    exclusive = False
    reverse = False
    name = "cumsum_9"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger array, axis=-1
    x = np.random.rand(5, 5, 5).astype(np.float32)
    axis = np.int32(-1)
    exclusive = False
    reverse = False
    name = "cumsum_10"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: different axis type
    x = np.array([1, 2, 3], dtype=np.int32)
    axis = np.int64(0)
    exclusive = False
    reverse = False
    name = "cumsum_11"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: uint32
    x = np.array([1, 2, 3], dtype=np.uint32)
    axis = np.int32(0)
    exclusive = False
    reverse = False
    name = "cumsum_12"
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Cumsum"] = tf_raw_ops_cumsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cumsum'.")

check_valid('tf.raw_ops.Cumsum', generated_inputs['tf.raw_ops.Cumsum'], lib="tf", suffix=0)
