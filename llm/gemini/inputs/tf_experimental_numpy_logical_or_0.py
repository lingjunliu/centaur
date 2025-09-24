
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_logical_or_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two boolean tensors
    x1 = tf.constant([True, False, True])
    x2 = tf.constant([False, True, True])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two dimensional tensors
    x1 = tf.constant([[True, False], [False, True]])
    x2 = tf.constant([[False, True], [True, False]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes, but broadcastable
    x1 = tf.constant([True, False])
    x2 = tf.constant([[False], [True]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three dimensional tensors
    x1 = tf.constant([[[True, False], [False, True]], [[False, True], [True, False]]])
    x2 = tf.constant([[[False, True], [True, False]], [[True, False], [False, True]]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using numpy arrays converted to tensors
    x1 = tf.constant(np.array([True, False, True], dtype=bool))
    x2 = tf.constant(np.array([False, True, False], dtype=bool))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with a scalar
    x1 = tf.constant([True, False, True])
    x2 = tf.constant(False)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting with a scalar
    x1 = tf.constant([True, False, True])
    x2 = tf.constant(True)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Two dimensional tensors with numpy arrays
    x1 = tf.constant(np.array([[True, False], [False, True]]))
    x2 = tf.constant(np.array([[False, True], [True, False]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: More complex shape
    x1 = tf.constant(np.random.choice([True, False], size=(2, 3, 4)))
    x2 = tf.constant(np.random.choice([True, False], size=(2, 3, 4)))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensors with shape (1, n)
    x1 = tf.constant(np.random.choice([True, False], size=(1, 5)))
    x2 = tf.constant(np.random.choice([True, False], size=(1, 5)))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.logical_or"] = tf_experimental_numpy_logical_or_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.logical_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.logical_or'.")

check_valid('tf.experimental.numpy.logical_or', generated_inputs['tf.experimental.numpy.logical_or'], lib="tf", suffix=0)
