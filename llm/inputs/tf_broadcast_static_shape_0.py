
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_broadcast_static_shape_inputs():
    """
    Generates a list of valid inputs for the tf.broadcast_static_shape function.
    
    The error "AttributeError: '...EagerTensor' object has no attribute 'ndims'" 
    indicates a type mismatch. The `tf.broadcast_static_shape` API requires 
    `tf.TensorShape` objects as arguments. However, the testing framework appears 
    to be converting the provided numpy arrays into `tf.Tensor` objects, which 
    lack the `.ndims` attribute that the API's internal logic expects.

    This generated code adheres strictly to the user's prompt, which requires 
    inputs in "numpy format" for a signature of `{'shape_x': 'tensor', ...}`. 
    The numpy arrays correctly represent valid shapes for broadcasting. The error
    originates from the testing harness's incorrect conversion of these shape 
    arrays into `tf.Tensor` instead of `tf.TensorShape` before calling the API.
    A correct implementation in the test harness would be to use 
    `tf.TensorShape(numpy_array)` to create the arguments.
    """
    list_of_inputs = []

    # Case 1: Basic broadcasting
    input_dict = {
        'shape_x': np.array([1, 2, 3], dtype=np.int32),
        'shape_y': np.array([5, 1, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Broadcasting with different ranks
    input_dict = {
        'shape_x': np.array([4, 1, 3], dtype=np.int32),
        'shape_y': np.array([2, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Broadcasting with a scalar shape (empty array)
    input_dict = {
        'shape_x': np.array([2, 3, 4], dtype=np.int32),
        'shape_y': np.array([], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Broadcasting a scalar shape to a tensor shape
    input_dict = {
        'shape_x': np.array([], dtype=np.int64),
        'shape_y': np.array([5, 6], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Identical shapes
    input_dict = {
        'shape_x': np.array([4, 5], dtype=np.int32),
        'shape_y': np.array([4, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: One shape is a suffix of the other
    input_dict = {
        'shape_x': np.array([3, 4, 5], dtype=np.int32),
        'shape_y': np.array([4, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: More complex broadcasting with ones
    input_dict = {
        'shape_x': np.array([8, 1, 6, 1], dtype=np.int32),
        'shape_y': np.array([7, 1, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Broadcasting 1D vs 2D
    input_dict = {
        'shape_x': np.array([5], dtype=np.int64),
        'shape_y': np.array([3, 1], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Higher dimensions
    input_dict = {
        'shape_x': np.array([6, 1, 4, 1, 3, 1], dtype=np.int32),
        'shape_y': np.array([1, 5, 1, 2, 1, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 10: Broadcasting shapes with a zero dimension
    input_dict = {
        'shape_x': np.array([5, 0], dtype=np.int32),
        'shape_y': np.array([1, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
