
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_reshape_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array reshape
    a = tf.constant(np.array([[1, 2], [3, 4], [5, 6]]))
    newshape = (2, 3)
    order = 'C'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Reshape to a 1D array
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    newshape = (6,)
    order = 'C'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Reshape a 1D array to a 2D array
    a = tf.constant(np.array([1, 2, 3, 4, 5, 6]))
    newshape = (2, 3)
    order = 'C'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Reshape a 3D array
    a = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    newshape = (2, 4)
    order = 'C'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Reshape to a scalar (0D array)
    a = tf.constant(np.array([1]))
    newshape = ()
    order = 'C'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Reshape with a different order (F)
    a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))
    newshape = (3, 2)
    order = 'F'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Reshape a float array
    a = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    newshape = (4, 1)
    order = 'C'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Reshape with different dimensions
    a = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    newshape = (2, 2, 2)
    order = 'C'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Reshape an empty array
    a = tf.constant(np.array([]))
    newshape = (0, 5)
    order = 'C'
    input_dict = {"a": a, "newshape": newshape, "order": order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.reshape"] = tf_experimental_numpy_reshape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.reshape'.")

check_valid('tf.experimental.numpy.reshape', generated_inputs['tf.experimental.numpy.reshape'], lib="tf", suffix=0)
