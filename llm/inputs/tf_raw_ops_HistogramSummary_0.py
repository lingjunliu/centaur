
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_HistogramSummary_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 histogram
    tag = tf.convert_to_tensor(np.array("float_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float32))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 histogram with negative values
    tag = tf.convert_to_tensor(np.array("int_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([-1, 0, 1, 2, -2, 1, 0], dtype=np.int32))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 histogram with a different name
    tag = tf.convert_to_tensor(np.array("double_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64))
    input_dict = {"tag": tag, "values": values, "name": "my_histogram"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: uint8 histogram
    tag = tf.convert_to_tensor(np.array("uint8_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([10, 20, 30, 40, 50], dtype=np.uint8))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int16 histogram
    tag = tf.convert_to_tensor(np.array("int16_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([-100, 0, 100, 200, -200], dtype=np.int16))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64 histogram
    tag = tf.convert_to_tensor(np.array("int64_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([-1000000000, 0, 1000000000, 2000000000], dtype=np.int64))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 histogram
    tag = tf.convert_to_tensor(np.array("bfloat16_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([1.0, 2.0, 3.0], dtype=np.float16))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint16 histogram
    tag = tf.convert_to_tensor(np.array("uint16_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([100, 200, 300, 400, 500], dtype=np.uint16))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: half histogram
    tag = tf.convert_to_tensor(np.array("half_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([1.0, 2.0, 3.0], dtype=np.float16, copy=False))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint32 histogram
    tag = tf.convert_to_tensor(np.array("uint32_histogram", dtype=np.string_))
    values = tf.convert_to_tensor(np.array([1000, 2000, 3000, 4000, 5000], dtype=np.uint32))
    input_dict = {"tag": tag, "values": values, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.HistogramSummary"] = tf_raw_ops_HistogramSummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.HistogramSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.HistogramSummary'.")

check_valid('tf.raw_ops.HistogramSummary', generated_inputs['tf.raw_ops.HistogramSummary'], lib="tf", suffix=0)
