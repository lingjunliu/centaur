
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DataFormatDimMap_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array(2, dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_1",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "map_case_2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[-4, -3], [-2, -1]], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_3",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[[-4, -3], [-2, -1]], [[0, 1], [2, 3]]], dtype=np.int64)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NCHW",
        "name": "map_case_4",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.zeros((7,), dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NHWC",
        "name": "map_case_5",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([
        [-4, -3, -2],
        [-1, 0, 1],
        [2, 3, -4],
        [-2, 1, 0]
    ], dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_6",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array(-4, dtype=np.int64)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "map_case_7",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[[-4, -1, 0, 3]]], dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_8",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    base = np.arange(25, dtype=np.int64)
    x = (base % 8) - 4
    x = x.reshape(5, 5)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "map_case_9",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[[-4, -3, -2]], [[-1, 0, 1]]]], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_10",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array([3, 2, 1, 0, -1, -2, -3, -4, -1, 0], dtype=np.int32)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NCHW",
        "name": "map_case_11",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    x = np.array([[-2], [3], [-4]], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NHWC",
        "name": "map_case_12",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DataFormatDimMap"] = tf_raw_ops_DataFormatDimMap_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DataFormatDimMap' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DataFormatDimMap'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DataFormatDimMap', generated_inputs['tf.raw_ops.DataFormatDimMap'], lib="tf", suffix=0)
