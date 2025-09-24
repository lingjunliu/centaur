
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_promote_types_inputs():
    list_of_inputs = []

    # Input 1
    type1 = np.dtype(np.int8)
    type2 = np.dtype(np.float32)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    type1 = np.dtype(np.uint16)
    type2 = np.dtype(np.complex64)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    type1 = np.dtype(np.bool_)
    type2 = np.dtype(np.int64)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    type1 = np.dtype(np.float16)
    type2 = np.dtype(np.float64)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    type1 = np.dtype(np.complex128)
    type2 = np.dtype(np.int32)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    type1 = np.dtype(np.uint8)
    type2 = np.dtype(np.float16)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    type1 = np.dtype(np.int16)
    type2 = np.dtype(np.complex64)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    type1 = np.dtype(np.float32)
    type2 = np.dtype(np.int8)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    type1 = np.dtype(np.float64)
    type2 = np.dtype(np.uint32)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    type1 = np.dtype(np.int32)
    type2 = np.dtype(np.complex128)
    input_dict = {"type1": type1, "type2": type2}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.promote_types"] = tf_promote_types_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.promote_types' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.promote_types'.")

check_valid('tf.experimental.numpy.promote_types', generated_inputs['tf.experimental.numpy.promote_types'], lib="tf", suffix=0)
