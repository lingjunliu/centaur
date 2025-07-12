
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dtypes_saturate_cast_inputs():
    list_of_inputs = []

    # Input 1
    value = np.array([-1.5, 0.0, 2.3, 5.7]).astype(np.float32)
    dtype = np.int32
    name = "saturate_cast_1"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.array([-100, 0, 100, 200]).astype(np.int64)
    dtype = np.int8
    name = "saturate_cast_2"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.array([[-1.0, 2.0], [3.0, -4.0]]).astype(np.float64)
    dtype = np.int16
    name = "saturate_cast_3"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.array([256, -257]).astype(np.int32)
    dtype = np.int8
    name = "saturate_cast_4"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.array([1.0, 0.0, -1.0]).astype(np.float32)
    dtype = np.uint8
    name = "saturate_cast_5"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.array([65536, -65537]).astype(np.int64)
    dtype = np.int16
    name = "saturate_cast_6"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.array([1.5, 2.5, 3.5]).astype(np.float32)
    dtype = np.int32
    name = "saturate_cast_7"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.array([[-1000.0, 1000.0], [-500.0, 500.0]]).astype(np.float32)
    dtype = np.int8
    name = "saturate_cast_8"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.array([0.0, 1.0, 2.0, 3.0]).astype(np.float64)
    dtype = np.uint8
    name = "saturate_cast_9"
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.array([-1, 0, 1]).astype(np.int32)
    dtype = np.float32
    name = None
    input_dict = {"value": value, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.dtypes.saturate_cast"] = tf_dtypes_saturate_cast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.dtypes.saturate_cast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.dtypes.saturate_cast'.")

check_valid('tf.dtypes.saturate_cast', generated_inputs['tf.dtypes.saturate_cast'], lib="tf", suffix=0)
