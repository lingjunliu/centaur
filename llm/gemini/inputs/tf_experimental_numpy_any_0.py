
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_any_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with axis=None, keepdims=False
    a = np.array([False, False, True]).astype(np.bool_)
    axis = None
    keepdims = False
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with axis=0, keepdims=False
    a = np.array([[False, True], [False, False]]).astype(np.bool_)
    axis = 0
    keepdims = False
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with axis=1, keepdims=True
    a = np.array([[False, True], [False, False]]).astype(np.bool_)
    axis = 1
    keepdims = True
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with axis=0, keepdims=False
    a = np.array([[[False, True], [True, False]], [[False, False], [True, True]]]).astype(np.bool_)
    axis = 0
    keepdims = False
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with axis=1, keepdims=True
    a = np.array([[[False, True], [True, False]], [[False, False], [True, True]]]).astype(np.bool_)
    axis = 1
    keepdims = True
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All False values, axis=None, keepdims=False
    a = np.array([False, False, False]).astype(np.bool_)
    axis = None
    keepdims = False
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty array
    a = np.array([]).astype(np.bool_)
    axis = None
    keepdims = False
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with a single True value
    a = np.array([True]).astype(np.bool_)
    axis = None
    keepdims = False
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array
    a = np.random.choice([False, True], size=(2, 3, 4, 5)).astype(np.bool_)
    axis = 2
    keepdims = True
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative axis, 2D array
    a = np.array([[False, True], [False, False]]).astype(np.bool_)
    axis = -1
    keepdims = False
    input_dict = {"a": tf.constant(a), "axis": axis, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.any"] = tf_experimental_numpy_any_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.any' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.any'.")

check_valid('tf.experimental.numpy.any', generated_inputs['tf.experimental.numpy.any'], lib="tf", suffix=0)
