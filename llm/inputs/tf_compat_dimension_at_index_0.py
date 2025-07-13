
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_compat_dimension_at_index_inputs():
    list_of_inputs = []

    # Input 1
    shape = tf.TensorShape([1, 2, 3])
    index = np.int32(0)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = tf.TensorShape([None, 2, 3])
    index = np.int32(0)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = tf.TensorShape([1, None, 3])
    index = np.int32(1)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = tf.TensorShape([1, 2, None])
    index = np.int32(2)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = tf.TensorShape([None, None, None])
    index = np.int32(0)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = tf.TensorShape([1])
    index = np.int32(0)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = tf.TensorShape([None])
    index = np.int32(0)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = tf.TensorShape([1, 2, 3, 4, 5])
    index = np.int32(4)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = tf.TensorShape([1, 2])
    index = np.int32(1)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = tf.TensorShape([None, 2])
    index = np.int32(1)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.compat.dimension_at_index"] = tf_compat_dimension_at_index_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.compat.dimension_at_index' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.dimension_at_index'.")

check_valid('tf.compat.dimension_at_index', generated_inputs['tf.compat.dimension_at_index'], lib="tf", suffix=0)
