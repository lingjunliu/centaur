
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fused_pad_conv2d_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1, 0, 1], [0, 1, 0], [1, 1, 1]], [[1, 0, 1], [0, 1, 0], [1, 1, 1]]]], dtype=np.float32)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    name_val = "fused_pad_conv2d_1"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]], dtype=np.float32)
    mode_val = "SYMMETRIC"
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    name_val = "fused_pad_conv2d_2"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=np.float32)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1], [1], [1], [1]], [[1], [1], [1], [1]]]], dtype=np.float32)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    name_val = "fused_pad_conv2d_3"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    paddings_val = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1], [1]], [[1], [1]]]], dtype=np.float32)
    mode_val = "SYMMETRIC"
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    name_val = "fused_pad_conv2d_4"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1], [1], [1]], [[1], [1], [1]]]], dtype=np.float32)
    mode_val = "REFLECT"
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    name_val = "fused_pad_conv2d_5"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float64)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]], dtype=np.float64)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    name_val = "fused_pad_conv2d_6"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float64)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]], dtype=np.float64)
    mode_val = "SYMMETRIC"
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    name_val = "fused_pad_conv2d_7"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_val = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=np.float64)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1], [1], [1], [1]], [[1], [1], [1], [1]]]], dtype=np.float64)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    name_val = "fused_pad_conv2d_8"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float64)
    paddings_val = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1], [1]], [[1], [1]]]], dtype=np.float64)
    mode_val = "SYMMETRIC"
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    name_val = "fused_pad_conv2d_9"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float64)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.array([[[[1], [1], [1]], [[1], [1], [1]]]], dtype=np.float64)
    mode_val = "REFLECT"
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    name_val = "fused_pad_conv2d_10"

    input_dict = {
        "input": input_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedPadConv2D"] = tf_raw_ops_fused_pad_conv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedPadConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedPadConv2D'.")

check_valid('tf.raw_ops.FusedPadConv2D', generated_inputs['tf.raw_ops.FusedPadConv2D'], lib="tf", suffix=0)
