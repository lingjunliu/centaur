
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedPadConv2D_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 1).astype(np.float32)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    name_str = "fused_pad_conv2d_1"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 5).astype(np.float32)
    paddings_tensor = np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(5, 5, 5, 2).astype(np.float32)
    mode_str = "SYMMETRIC"
    strides_list = [1, 2, 2, 1]
    padding_str = "SAME"
    name_str = "fused_pad_conv2d_2"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(4, 7, 7, 1).astype(np.float32)
    paddings_tensor = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(1, 1, 1, 4).astype(np.float32)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    name_str = "fused_pad_conv2d_3"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 12, 12, 8).astype(np.float16)
    paddings_tensor = np.array([[0, 0], [3, 3], [3, 3], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(7, 7, 8, 3).astype(np.float16)
    mode_str = "SYMMETRIC"
    strides_list = [1, 3, 3, 1]
    padding_str = "SAME"
    name_str = "fused_pad_conv2d_4"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(2, 6, 6, 4).astype(np.float64)
    paddings_tensor = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(2, 2, 4, 2).astype(np.float64)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    name_str = "fused_pad_conv2d_5"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 4, 4, 2).astype(np.float32)
    paddings_tensor = np.array([[0, 0], [2, 1], [1, 2], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 2, 1).astype(np.float32)
    mode_str = "SYMMETRIC"
    strides_list = [1, 2, 2, 1]
    padding_str = "SAME"
    name_str = "fused_pad_conv2d_6"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(3, 8, 8, 6).astype(np.float32)
    paddings_tensor = np.array([[0, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(2, 2, 6, 3).astype(np.float32)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    name_str = "fused_pad_conv2d_7"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    input_tensor = np.random.rand(1, 16, 16, 1).astype(np.float32)
    paddings_tensor = np.array([[0, 0], [4, 4], [4, 4], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(9, 9, 1, 1).astype(np.float32)
    mode_str = "SYMMETRIC"
    strides_list = [1, 4, 4, 1]
    padding_str = "SAME"
    name_str = "fused_pad_conv2d_8"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(2, 3, 3, 2).astype(np.float16)
    paddings_tensor = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(1, 1, 2, 1).astype(np.float16)
    mode_str = "REFLECT"
    strides_list = [1, 1, 1, 1]
    padding_str = "VALID"
    name_str = "fused_pad_conv2d_9"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 28, 28, 3).astype(np.float32)
    paddings_tensor = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_tensor = np.random.rand(3, 3, 3, 32).astype(np.float32)
    mode_str = "SYMMETRIC"
    strides_list = [1, 1, 1, 1]
    padding_str = "SAME"
    name_str = "fused_pad_conv2d_10"
    input_dict = {"input": input_tensor, "paddings": paddings_tensor, "filter": filter_tensor, "mode": mode_str, "strides": strides_list, "padding": padding_str, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_FusedPadConv2D_inputs()
generated_inputs["tf.raw_ops.FusedPadConv2D"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.FusedPadConv2D"].append({
        "input": input_dict["input"],
        "paddings": input_dict["paddings"],
        "filter": input_dict["filter"],
        "mode": input_dict["mode"],
        "strides": input_dict["strides"],
        "padding": input_dict["padding"],
        "name": input_dict["name"]
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FusedPadConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FusedPadConv2D'.")

check_valid('tf.raw_ops.FusedPadConv2D', generated_inputs['tf.raw_ops.FusedPadConv2D'], lib="tf", suffix=0)
