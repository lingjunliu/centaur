
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_compat_dimension_at_index_inputs():
    list_of_inputs = []

    # Input 1: Simple case
    shape = tf.TensorShape([1, 2, 3])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Index in the middle
    shape = tf.TensorShape([1, 2, 3])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Last index
    shape = tf.TensorShape([1, 2, 3])
    index = 2
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rank 1 shape
    shape = tf.TensorShape([5])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rank 4 shape
    shape = tf.TensorShape([1, 2, 3, 4])
    index = 2
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Unknown dimension
    shape = tf.TensorShape([None, 2, 3])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape with unknown and known dimensions
    shape = tf.TensorShape([1, None, 3])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another valid rank 4 input
    shape = tf.TensorShape([5,6,7,8])
    index = 3
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Index 0 for unknown shape, valid
    shape = tf.TensorShape(None)
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 2 shape with unknown dimension at the beginning
    shape = tf.TensorShape([None, 5])
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
