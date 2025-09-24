
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_placeholder_with_default_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.array(1, dtype=np.int32)
    shape_val = []
    name_val = "placeholder_1"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.array([1, 2, 3], dtype=np.float32)
    shape_val = [3]
    name_val = "placeholder_2"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.array([[1, 2], [3, 4]], dtype=np.int64)
    shape_val = [2, 2]
    name_val = "placeholder_3"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    shape_val = [2, 2, 2]
    name_val = "placeholder_4"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = np.array(-1, dtype=np.int32)
    shape_val = []
    name_val = "placeholder_5"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.array([-1, -2, -3], dtype=np.float32)
    shape_val = [3]
    name_val = "placeholder_6"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = np.array([[-1, -2], [-3, -4]], dtype=np.int64)
    shape_val = [2, 2]
    name_val = "placeholder_7"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_val = np.array([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]], dtype=np.float64)
    shape_val = [2, 2, 2]
    name_val = "placeholder_8"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.array(0, dtype=np.int32)
    shape_val = []
    name_val = "placeholder_9"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.array([0, 0, 0], dtype=np.float32)
    shape_val = [3]
    name_val = "placeholder_10"
    input_dict = {"input": input_val, "shape": shape_val, "name": name_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.PlaceholderWithDefault"] = tf_raw_ops_placeholder_with_default_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PlaceholderWithDefault' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PlaceholderWithDefault'.")

check_valid('tf.raw_ops.PlaceholderWithDefault', generated_inputs['tf.raw_ops.PlaceholderWithDefault'], lib="tf", suffix=0)
