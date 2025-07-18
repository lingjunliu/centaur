
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy


def tf_compat_dimension_at_index_inputs():
    list_of_inputs = []

    # Input 1: 2D shape, positive index 0
    input_dict = {
        'shape': tf.TensorShape([3, 2]),
        'index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D shape, positive index 1
    input_dict = {
        'shape': tf.TensorShape([3, 2]),
        'index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D shape, negative index
    input_dict = {
        'shape': tf.TensorShape([4, 5, 6]),
        'index': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape, another negative index
    input_dict = {
        'shape': tf.TensorShape([4, 5, 6]),
        'index': -3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D vector shape
    input_dict = {
        'shape': tf.TensorShape([10]),
        'index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: High-rank shape
    input_dict = {
        'shape': tf.TensorShape([1, 2, 3, 4, 5, 6]),
        'index': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape with a zero dimension
    input_dict = {
        'shape': tf.TensorShape([5, 0, 10]),
        'index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Shape with an unknown dimension
    input_dict = {
        'shape': tf.TensorShape([2, None, 4]),
        'index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Shape with an unknown dimension, accessing a known one
    input_dict = {
        'shape': tf.TensorShape([2, None, 4]),
        'index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Shape with unknown rank
    input_dict = {
        'shape': tf.TensorShape(None),
        'index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.compat.dimension_at_index"] = tf_compat_dimension_at_index_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.compat.dimension_at_index' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.dimension_at_index'.")

check_valid('tf.compat.dimension_at_index', generated_inputs['tf.compat.dimension_at_index'], lib="tf", suffix=0)
