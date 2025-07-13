
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_less_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 comparison
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 comparison with negative values
    x = np.array([-1, 0, 1], dtype=np.int32)
    y = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 comparison with different shapes (broadcasting)
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int64 comparison
    x = np.array([5, 4, 6], dtype=np.int64)
    y = np.array([5, 6, 6], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "int64_comp"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Uint8 comparison
    x = np.array([255, 0, 128], dtype=np.uint8)
    y = np.array([128, 128, 128], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Int16 comparison
    x = np.array([-32768, 0, 32767], dtype=np.int16)
    y = np.array([0, 0, 0], dtype=np.int16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Uint16 comparison
    x = np.array([0, 30000, 65535], dtype=np.uint16)
    y = np.array([30000, 30000, 30000], dtype=np.uint16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Half comparison (float16)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([2.0, 2.0, 2.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Uint32 Comparison
    x = np.array([10, 20, 30], dtype=np.uint32)
    y = np.array([15, 15, 35], dtype=np.uint32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float32 with name
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "float_comp"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.LessEqual"] = tf_raw_ops_less_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.LessEqual' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LessEqual'.")

check_valid('tf.raw_ops.LessEqual', generated_inputs['tf.raw_ops.LessEqual'], lib="tf", suffix=0)
