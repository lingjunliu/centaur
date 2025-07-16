
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DataFormatDimMap_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    src_format = "NHWC"
    dst_format = "NCHW"
    name = "data_format_dim_map_1"

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1, -2, -3, -4], dtype=np.int32)
    src_format = "NCHW"
    dst_format = "NHWC"
    name = "data_format_dim_map_2"

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([0, 1, 2, 3], dtype=np.int64)
    src_format = "HWCN"
    dst_format = "CHWN"
    name = "data_format_dim_map_3"

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([-1, -2, -3, -4], dtype=np.int64)
    src_format = "CHWN"
    dst_format = "HWCN"
    name = "data_format_dim_map_4"

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    src_format = "NHWC"
    dst_format = "NCHW"
    name = None

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([-1, -2, -3, -4], dtype=np.int64)
    src_format = "NCHW"
    dst_format = "NHWC"
    name = ""

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    src_format = "HWNC"
    dst_format = "NWHC"
    name = "data_format_dim_map_7"

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    x = np.array([-1, -2, -3, -4], dtype=np.int64)
    src_format = "HWNC"
    dst_format = "NWHC"
    name = "data_format_dim_map_8"

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    src_format = "NCWH"
    dst_format = "WHCN"
    name = "data_format_dim_map_9"

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([-1, -2, -3, -4], dtype=np.int64)
    src_format = "WHCN"
    dst_format = "NCWH"
    name = "data_format_dim_map_10"

    input_dict = {
        "x": x,
        "src_format": src_format,
        "dst_format": dst_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DataFormatDimMap"] = tf_raw_ops_DataFormatDimMap_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DataFormatDimMap' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DataFormatDimMap'.")

check_valid('tf.raw_ops.DataFormatDimMap', generated_inputs['tf.raw_ops.DataFormatDimMap'], lib="tf", suffix=0)
