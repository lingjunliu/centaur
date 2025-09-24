
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_dtypes_as_dtype_inputs():
    list_of_inputs = []

    # Input 1: numpy.int8
    input_dict = {"type_value": np.int8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: numpy.int16
    input_dict = {"type_value": np.int16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: numpy.int32
    input_dict = {"type_value": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: numpy.int64
    input_dict = {"type_value": np.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy.uint8
    input_dict = {"type_value": np.uint8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy.uint16
    input_dict = {"type_value": np.uint16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy.uint32
    input_dict = {"type_value": np.uint32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy.uint64
    input_dict = {"type_value": np.uint64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: numpy.float16
    input_dict = {"type_value": np.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy.float32
    input_dict = {"type_value": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.dtypes.as_dtype"] = tf_dtypes_as_dtype_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.dtypes.as_dtype' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.dtypes.as_dtype'.")

check_valid('tf.dtypes.as_dtype', generated_inputs['tf.dtypes.as_dtype'], lib="tf", suffix=0)
