
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_iinfo_inputs():
    list_of_inputs = []

    # Input 1: np.int8
    input_dict = {"int_type": np.int8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: np.int16
    input_dict = {"int_type": np.int16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: np.int32
    input_dict = {"int_type": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.int64
    input_dict = {"int_type": np.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: np.uint8
    input_dict = {"int_type": np.uint8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.uint16
    input_dict = {"int_type": np.uint16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.uint32
    input_dict = {"int_type": np.uint32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.uint64
    input_dict = {"int_type": np.uint64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: tf.int8 to numpy
    input_dict = {"int_type": tf.as_dtype(tf.int8).as_numpy_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: tf.uint16 to numpy
    input_dict = {"int_type": tf.as_dtype(tf.uint16).as_numpy_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.iinfo"] = tf_experimental_numpy_iinfo_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.iinfo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.iinfo'.")

check_valid('tf.experimental.numpy.iinfo', generated_inputs['tf.experimental.numpy.iinfo'], lib="tf", suffix=0)
