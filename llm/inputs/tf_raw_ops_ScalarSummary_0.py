
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScalarSummary_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    tags = np.array([b"tag1"], dtype="S4")
    values = np.array([1.0], dtype=np.float32)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple tags and values
    tags = np.array([b"tag1", b"tag2", b"tag3"], dtype="S4")
    values = np.array([1.0, 2.5, -0.7], dtype=np.float32)
    name = "my_summary"
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int64 values
    tags = np.array([b"tag1"], dtype="S4")
    values = np.array([10000000000], dtype=np.int64)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different dtype (float64)
    tags = np.array([b"tag1"], dtype="S4")
    values = np.array([3.14159], dtype=np.float64)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    tags = np.array([b"tag1", b"tag2"], dtype="S4")
    values = np.array([-1.0, -2.0], dtype=np.float32)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty name
    tags = np.array([b"tag1"], dtype="S4")
    values = np.array([1.0], dtype=np.float32)
    name = ""
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7:  int32 values
    tags = np.array([b"tag1"], dtype="S4")
    values = np.array([100], dtype=np.int32)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  uint8 values
    tags = np.array([b"tag1"], dtype="S4")
    values = np.array([255], dtype=np.uint8)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple values with int type
    tags = np.array([b"tag1", b"tag2", b"tag3"], dtype="S4")
    values = np.array([1, 2, 3], dtype=np.int32)
    name = "int_summary"
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half
    tags = np.array([b"tag1"], dtype="S4")
    values = np.array([1.0], dtype=np.float16)
    name = None
    input_dict = {"tags": tags, "values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScalarSummary"] = tf_raw_ops_ScalarSummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScalarSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScalarSummary'.")

check_valid('tf.raw_ops.ScalarSummary', generated_inputs['tf.raw_ops.ScalarSummary'], lib="tf", suffix=0)
