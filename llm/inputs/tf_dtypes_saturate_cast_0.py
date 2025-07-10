
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dtypes_saturate_cast_inputs():
    list_of_inputs = []

    # Input 1: Basic cast from float32 to int32
    value = tf.constant(np.array([1.5, 2.7, 3.9], dtype=np.float32))
    dtype = tf.int32
    name = "float_to_int"
    input_dict = {"value": value.numpy(), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Cast from int64 to float16
    value = tf.constant(np.array([1000000000, 2000000000, 3000000000], dtype=np.int64))
    dtype = tf.float16
    name = "int64_to_float16"
    input_dict = {"value": value.numpy(), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Cast from int8 to uint8 (with negative values)
    value = tf.constant(np.array([-10, 0, 10, 127], dtype=np.int8))
    dtype = tf.uint8
    name = "int8_to_uint8"
    input_dict = {"value": value.numpy(), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Cast from float64 to int8 (with values exceeding limits)
    value = tf.constant(np.array([-200.0, 0.0, 200.0], dtype=np.float64))
    dtype = tf.int8
    name = "float64_to_int8"
    input_dict = {"value": value.numpy(), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Cast from uint16 to int16 (with values exceeding limits)
    value = tf.constant(np.array([0, 32767, 65000], dtype=np.uint16))
    dtype = tf.int16
    name = "uint16_to_int16"
    input_dict = {"value": value.numpy(), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Cast from float32 to bfloat16
    value = tf.constant(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    dtype = tf.bfloat16
    name = "float32_to_bfloat16"
    input_dict = {"value": value.numpy(), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional tensor from int32 to float32
    value = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32))
    dtype = tf.float32
    name = "int32_to_float32_multi"
    input_dict = {"value": value.numpy(), "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Cast from float32 to int32 with negative and positive large values
    value = tf.constant(np.array([-2147483647.0, 0.0, 2147483647.0], dtype=np.float32))
    dtype = tf.int32
    name = "float32_to_int32_limits"
    input_dict = {"value": value.numpy(), "dtype": dtype, "name": name}
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
