
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedResizeAndPadConv2D_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.rand(1, 28, 28, 3).astype(np.float32)
    size_val = np.array([56, 56], dtype=np.int32)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 16).astype(np.float32)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    resize_align_corners_val = False
    name_val = "fused_resize_pad_conv2d_1"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.random.rand(4, 14, 14, 1).astype(np.float32)
    size_val = np.array([28, 28], dtype=np.int32)
    paddings_val = np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(5, 5, 1, 8).astype(np.float32)
    mode_val = "SYMMETRIC"
    strides_val = [1, 2, 2, 1]
    padding_val = "VALID"
    resize_align_corners_val = True
    name_val = "fused_resize_pad_conv2d_2"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.random.rand(2, 10, 10, 3).astype(np.float32)
    size_val = np.array([20, 20], dtype=np.int32)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(2, 2, 3, 4).astype(np.float32)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    resize_align_corners_val = False
    name_val = "fused_resize_pad_conv2d_3"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    input_val = np.random.rand(1, 32, 32, 1).astype(np.float64)
    size_val = np.array([64, 64], dtype=np.int32)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 1, 1).astype(np.float64)
    mode_val = "SYMMETRIC"
    strides_val = [1, 2, 2, 1]
    padding_val = "VALID"
    resize_align_corners_val = True
    name_val = "fused_resize_pad_conv2d_4"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = np.random.rand(4, 8, 8, 3).astype(np.float32)
    size_val = np.array([16, 16], dtype=np.int32)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 2).astype(np.float32)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    resize_align_corners_val = False
    name_val = "fused_resize_pad_conv2d_5"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.random.rand(1, 64, 64, 1).astype(np.float32)
    size_val = np.array([32, 32], dtype=np.int32)
    paddings_val = np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(5, 5, 1, 1).astype(np.float32)
    mode_val = "SYMMETRIC"
    strides_val = [1, 4, 4, 1]
    padding_val = "VALID"
    resize_align_corners_val = True
    name_val = "fused_resize_pad_conv2d_6"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_val = np.random.rand(2, 5, 5, 3).astype(np.float32)
    size_val = np.array([10, 10], dtype=np.int32)
    paddings_val = np.array([[0, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(2, 2, 3, 4).astype(np.float32)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    resize_align_corners_val = False
    name_val = "fused_resize_pad_conv2d_7"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_val = np.random.rand(1, 16, 16, 1).astype(np.float64)
    size_val = np.array([8, 8], dtype=np.int32)
    paddings_val = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 1, 1).astype(np.float64)
    mode_val = "SYMMETRIC"
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    resize_align_corners_val = True
    name_val = "fused_resize_pad_conv2d_8"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.random.rand(4, 4, 4, 3).astype(np.float32)
    size_val = np.array([4, 4], dtype=np.int32)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(1, 1, 3, 2).astype(np.float32)
    mode_val = "REFLECT"
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    resize_align_corners_val = False
    name_val = "fused_resize_pad_conv2d_9"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.random.rand(1, 32, 32, 1).astype(np.float32)
    size_val = np.array([16, 16], dtype=np.int32)
    paddings_val = np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 1, 1).astype(np.float32)
    mode_val = "SYMMETRIC"
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    resize_align_corners_val = True
    name_val = "fused_resize_pad_conv2d_10"

    input_dict = {
        "input": input_val,
        "size": size_val,
        "paddings": paddings_val,
        "filter": filter_val,
        "mode": mode_val,
        "strides": strides_val,
        "padding": padding_val,
        "resize_align_corners": resize_align_corners_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedResizeAndPadConv2D"] = tf_raw_ops_FusedResizeAndPadConv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedResizeAndPadConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedResizeAndPadConv2D'.")

check_valid('tf.raw_ops.FusedResizeAndPadConv2D', generated_inputs['tf.raw_ops.FusedResizeAndPadConv2D'], lib="tf", suffix=0)
