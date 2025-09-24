
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fusedresizeandpadconv2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size_tensor = np.array([64, 64], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 16).astype(np.float32)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    resize_align_corners_bool = False
    name_str = "fused_resize_pad_conv2d_1"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(4, 16, 16, 8).astype(np.float32)
    size_tensor = np.array([32, 32], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(5, 5, 8, 32).astype(np.float32)
    mode_str = "SYMMETRIC"
    strides_list = [1, 2, 2, 1]
    padding_str = "SAME"
    resize_align_corners_bool = True
    name_str = "fused_resize_pad_conv2d_2"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 64, 64, 1).astype(np.float32)
    size_tensor = np.array([128, 128], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(1, 1, 1, 4).astype(np.float32)
    mode_str = "REFLECT"
    strides_list = [1, 4, 4, 1]
    padding_str = "VALID"
    resize_align_corners_bool = False
    name_str = "fused_resize_pad_conv2d_3"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 8, 8, 4).astype(np.float32)
    size_tensor = np.array([16, 16], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [3, 3], [3, 3], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(7, 7, 4, 8).astype(np.float32)
    mode_str = "SYMMETRIC"
    strides_list = [1, 1, 1, 1]
    padding_str = "SAME"
    resize_align_corners_bool = True
    name_str = "fused_resize_pad_conv2d_4"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5, half
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float16)
    size_tensor = np.array([64, 64], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 16).astype(np.float16)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    resize_align_corners_bool = False
    name_str = "fused_resize_pad_conv2d_5"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, double
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float64)
    size_tensor = np.array([64, 64], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 16).astype(np.float64)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    resize_align_corners_bool = False
    name_str = "fused_resize_pad_conv2d_6"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size_tensor = np.array([64, 64], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 16).astype(np.float32)
    mode_str = "REFLECT"
    strides_list = [1, 1, 2, 1]
    padding_str = "VALID"
    resize_align_corners_bool = False
    name_str = "fused_resize_pad_conv2d_7"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size_tensor = np.array([64, 64], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 16).astype(np.float32)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "SAME"
    resize_align_corners_bool = False
    name_str = "fused_resize_pad_conv2d_8"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size_tensor = np.array([64, 64], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 16).astype(np.float32)
    mode_str = "SYMMETRIC"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    resize_align_corners_bool = False
    name_str = "fused_resize_pad_conv2d_9"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 32, 32, 3).astype(np.float32)
    size_tensor = np.array([64, 64], dtype=np.int32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 16).astype(np.float32)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    resize_align_corners_bool = True
    name_str = "fused_resize_pad_conv2d_10"

    input_dict = {
        "input": input_tensor,
        "size": size_tensor,
        "paddings": paddings_tensor,
        "filter": filter_tensor,
        "mode": mode_str,
        "strides": strides_list,
        "padding": padding_str,
        "resize_align_corners": resize_align_corners_bool,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FusedResizeAndPadConv2D"] = tf_raw_ops_fusedresizeandpadconv2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedResizeAndPadConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedResizeAndPadConv2D'.")

check_valid('tf.raw_ops.FusedResizeAndPadConv2D', generated_inputs['tf.raw_ops.FusedResizeAndPadConv2D'], lib="tf", suffix=0)
