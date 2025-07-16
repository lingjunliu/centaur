
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv3DBackpropInputV2_inputs():
    list_of_inputs = []

    # Input 1
    input_sizes = np.array([1, 3, 32, 32, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 16).astype(np.float32)
    out_backprop_val = np.random.rand(1, 1, 30, 30, 16).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_input_1"
    input_dict = {"input_sizes": input_sizes, "filter": filter_val, "out_backprop": out_backprop_val, "strides": strides_val, "padding": padding_val, "data_format": data_format_val, "dilations": dilations_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_sizes = np.array([4, 5, 16, 16, 1], dtype=np.int32)
    filter_val = np.random.rand(5, 5, 5, 1, 8).astype(np.float32)
    out_backprop_val = np.random.rand(4, 1, 12, 12, 8).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_input_2"
    input_dict = {"input_sizes": input_sizes, "filter": filter_val, "out_backprop": out_backprop_val, "strides": strides_val, "padding": padding_val, "data_format": data_format_val, "dilations": dilations_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_sizes = np.array([2, 7, 64, 64, 3], dtype=np.int32)
    filter_val = np.random.rand(7, 7, 7, 3, 32).astype(np.float32)
    out_backprop_val = np.random.rand(2, 1, 64, 64, 32).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "SAME"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_input_3"
    input_dict = {"input_sizes": input_sizes, "filter": filter_val, "out_backprop": out_backprop_val, "strides": strides_val, "padding": padding_val, "data_format": data_format_val, "dilations": dilations_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_sizes = np.array([1, 3, 32, 32, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 16).astype(np.float32)
    out_backprop_val = np.random.rand(1, 1, 32, 32, 16).astype(np.float32)
    strides_val = [1, 1, 1, 1, 1]
    padding_val = "SAME"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_input_4"
    input_dict = {"input_sizes": input_sizes, "filter": filter_val, "out_backprop": out_backprop_val, "strides": strides_val, "padding": padding_val, "data_format": data_format_val, "dilations": dilations_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_sizes = np.array([1, 3, 32, 32, 3], dtype=np.int32)
    filter_val = np.random.rand(3, 3, 3, 3, 16).astype(np.float32)
    out_backprop_val = np.random.rand(1, 1, 16, 16, 16).astype(np.float32)
    strides_val = [1, 2, 2, 2, 1]
    padding_val = "VALID"
    data_format_val = "NDHWC"
    dilations_val = [1, 1, 1, 1, 1]
    name_val = "conv3d_backprop_input_5"
    input_dict = {"input_sizes": input_sizes, "filter": filter_val, "out_backprop": out_backprop_val, "strides": strides_val, "padding": padding_val, "data_format": data_format_val, "dilations": dilations_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_Conv3DBackpropInputV2_inputs()
generated_inputs["tf.raw_ops.Conv3DBackpropInputV2"] = []
for input_dict in inputs:
  generated_inputs["tf.raw_ops.Conv3DBackpropInputV2"].append({
      "input_sizes": tf.convert_to_tensor(input_dict["input_sizes"], dtype=tf.int32),
      "filter": tf.convert_to_tensor(input_dict["filter"], dtype=tf.float32),
      "out_backprop": tf.convert_to_tensor(input_dict["out_backprop"], dtype=tf.float32),
      "strides": input_dict["strides"],
      "padding": input_dict["padding"],
      "data_format": input_dict["data_format"],
      "dilations": input_dict["dilations"],
      "name": input_dict["name"]
  })
tf.experimental.numpy.experimental_enable_numpy_behavior()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Conv3DBackpropInputV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Conv3DBackpropInputV2'.")

check_valid('tf.raw_ops.Conv3DBackpropInputV2', generated_inputs['tf.raw_ops.Conv3DBackpropInputV2'], lib="tf", suffix=0)
