
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_compat_dimension_at_index_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a defined shape
    shape = tf.TensorShape([10, 20, 30])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Index in the middle
    shape = tf.TensorShape([10, 20, 30])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Last index
    shape = tf.TensorShape([10, 20, 30])
    index = 2
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shape with only one dimension
    shape = tf.TensorShape([5])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shape with unknown dimensions (None) at various indices
    shape = tf.TensorShape([None, 20, None])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another shape with unknown dimensions (None) at various indices
    shape = tf.TensorShape([10, None, 30])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Index 0 with partially defined shape
    shape = tf.TensorShape([None, 20, 30])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Index 1 with partially defined shape
    shape = tf.TensorShape([10, None, 30])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Index 2 with partially defined shape
    shape = tf.TensorShape([10, 20, None])
    index = 2
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Shape with no dimensions
    shape = tf.TensorShape([])
    index = 0
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
