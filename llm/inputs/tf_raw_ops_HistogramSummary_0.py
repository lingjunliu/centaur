
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_histogram_summary_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 histogram
    tag = np.array("float_histogram", dtype=np.bytes_)
    values = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float32)
    name = "histogram_1"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer histogram
    tag = np.array("int_histogram", dtype=np.bytes_)
    values = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=np.int32)
    name = "histogram_2"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 histogram with negative values
    tag = np.array("negative_float_histogram", dtype=np.bytes_)
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    name = "histogram_3"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty histogram
    tag = np.array("empty_histogram", dtype=np.bytes_)
    values = np.array([], dtype=np.float32)
    name = "histogram_4"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional histogram (2D)
    tag = np.array("2d_histogram", dtype=np.bytes_)
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    name = "histogram_5"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 histogram
    tag = np.array("uint8_histogram", dtype=np.bytes_)
    values = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
    name = "histogram_7"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 histogram with large values
    tag = np.array("int64_histogram", dtype=np.bytes_)
    values = np.array([10000000000, 20000000000, 30000000000], dtype=np.int64)
    name = "histogram_8"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16 histogram
    tag = np.array("bfloat16_histogram", dtype=np.bytes_)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    name = "histogram_9"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half (float16) histogram
    tag = np.array("half_histogram", dtype=np.bytes_)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    name = "histogram_10"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint32 histogram
    tag = np.array("uint32_histogram", dtype=np.bytes_)
    values = np.array([1000, 2000, 3000], dtype=np.uint32)
    name = "histogram_11"
    input_dict = {"tag": tag, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.HistogramSummary"] = tf_raw_ops_histogram_summary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.HistogramSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.HistogramSummary'.")

check_valid('tf.raw_ops.HistogramSummary', generated_inputs['tf.raw_ops.HistogramSummary'], lib="tf", suffix=0)
