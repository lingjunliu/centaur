
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def get_tf_broadcast_static_shape_inputs():
    """
    Generates a list of valid inputs for tf.broadcast_static_shape.
    The inputs must be tf.TensorShape objects as per the API documentation to resolve
    the 'EagerTensor' object has no attribute 'ndims' error.
    """
    list_of_inputs = []

    # Input 1: Basic case from documentation
    shape_x_1 = tf.TensorShape([1, 2, 3])
    shape_y_1 = tf.TensorShape([5, 1, 3])
    input_dict_1 = {'shape_x': shape_x_1, 'shape_y': shape_y_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Broadcasting shapes with different ranks
    shape_x_2 = tf.TensorShape([3])
    shape_y_2 = tf.TensorShape([2, 3])
    input_dict_2 = {'shape_x': shape_x_2, 'shape_y': shape_y_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Broadcasting where one dimension is 1
    shape_x_3 = tf.TensorShape([5, 1, 7])
    shape_y_3 = tf.TensorShape([1, 6, 7])
    input_dict_3 = {'shape_x': shape_x_3, 'shape_y': shape_y_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Higher dimensional broadcasting
    shape_x_4 = tf.TensorShape([8, 1, 6, 1])
    shape_y_4 = tf.TensorShape([7, 1, 5])
    input_dict_4 = {'shape_x': shape_x_4, 'shape_y': shape_y_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Broadcasting a scalar (empty shape) to a vector
    shape_x_5 = tf.TensorShape([])
    shape_y_5 = tf.TensorShape([5])
    input_dict_5 = {'shape_x': shape_x_5, 'shape_y': shape_y_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Identical shapes
    shape_x_6 = tf.TensorShape([5, 6, 7])
    shape_y_6 = tf.TensorShape([5, 6, 7])
    input_dict_6 = {'shape_x': shape_x_6, 'shape_y': shape_y_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: More complex rank difference
    shape_x_7 = tf.TensorShape([1, 5])
    shape_y_7 = tf.TensorShape([4, 3, 1, 1])
    input_dict_7 = {'shape_x': shape_x_7, 'shape_y': shape_y_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Broadcasting with a zero dimension
    shape_x_8 = tf.TensorShape([0, 5])
    shape_y_8 = tf.TensorShape([1, 5])
    input_dict_8 = {'shape_x': shape_x_8, 'shape_y': shape_y_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs["tf.broadcast_static_shape"] = get_tf_broadcast_static_shape_inputs()

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
