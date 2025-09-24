
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_static_shape_inputs():
    """
    Generates a list of valid inputs for the tf.broadcast_static_shape function.
    The API requires tf.TensorShape objects to be passed, not tf.Tensor objects.
    Providing tensors results in an `AttributeError` because EagerTensors lack
    the 'ndims' attribute that the function expects. This implementation provides
    the correct tf.TensorShape type to resolve the error.
    """
    list_of_inputs = []

    # Input 1: Classic example from documentation
    input_dict_1 = {
        'shape_x': tf.TensorShape([1, 2, 3]),
        'shape_y': tf.TensorShape([5, 1, 3])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Identical shapes
    input_dict_2 = {
        'shape_x': tf.TensorShape([4, 5]),
        'shape_y': tf.TensorShape([4, 5])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Broadcasting with a scalar-like shape [1]
    input_dict_3 = {
        'shape_x': tf.TensorShape([2, 3, 4]),
        'shape_y': tf.TensorShape([1])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Broadcasting with an empty shape (representing a 0-D tensor's shape)
    input_dict_4 = {
        'shape_x': tf.TensorShape([]),
        'shape_y': tf.TensorShape([5, 6])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Different number of dimensions where broadcasting is possible
    input_dict_5 = {
        'shape_x': tf.TensorShape([7, 8]),
        'shape_y': tf.TensorShape([3, 1, 8])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Different number of dimensions (other way)
    input_dict_6 = {
        'shape_x': tf.TensorShape([6, 1, 9]),
        'shape_y': tf.TensorShape([9])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Broadcasting with multiple 1s
    input_dict_7 = {
        'shape_x': tf.TensorShape([1, 2, 1, 4]),
        'shape_y': tf.TensorShape([5, 1, 3, 1])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Broadcasting a shape with a zero dimension
    input_dict_8 = {
        'shape_x': tf.TensorShape([5, 0]),
        'shape_y': tf.TensorShape([1, 0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Higher-rank shape with a lower-rank one
    input_dict_9 = {
        'shape_x': tf.TensorShape([2, 3, 4, 5]),
        'shape_y': tf.TensorShape([4, 5])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: More complex broadcasting with multiple 1s and different ranks
    input_dict_10 = {
        'shape_x': tf.TensorShape([8, 1, 6, 1]),
        'shape_y': tf.TensorShape([7, 1, 5])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.broadcast_static_shape"] = tf_broadcast_static_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.broadcast_static_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.broadcast_static_shape'.")

check_valid('tf.broadcast_static_shape', generated_inputs['tf.broadcast_static_shape'], lib="tf", suffix=0)
