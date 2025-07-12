
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_compat_dimension_at_index_inputs():
    list_of_inputs = []

    # Input 1: Simple valid case
    shape = tf.TensorShape([10, 20, 30])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another valid case with different index
    shape = tf.TensorShape([10, 20, 30])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using the last index
    shape = tf.TensorShape([10, 20, 30])
    index = 2
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Shape with only one dimension
    shape = tf.TensorShape([10])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shape with unknown dimensions
    shape = tf.TensorShape([None, 20, None])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Shape with unknown dimensions and different index
    shape = tf.TensorShape([None, 20, None])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Shape with a dimension of 1
    shape = tf.TensorShape([1])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Shape with many dimensions
    shape = tf.TensorShape([1, 2, 3, 4, 5, 6, 7, 8, 9])
    index = 5
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another shape with many dimensions and different index
    shape = tf.TensorShape([1, 2, 3, 4, 5, 6, 7, 8, 9])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Shape with a single unknown dimension
    shape = tf.TensorShape([None])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Shape with mixed known and unknown dimensions
    shape = tf.TensorShape([10, None, 30, None])
    index = 1
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
