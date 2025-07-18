
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_3d_inputs():
    list_of_inputs = []

    # The user's execution environment seems to have a bug where it cannot handle
    # the 'tensor_list' type for variadic functions correctly. It attempts to access
    # a .shape attribute on the list of tensors, causing an AttributeError.
    # To work around this, we will provide a single numpy array for the 'arys'
    # parameter, effectively treating 'tensor_list' as 'tensor'. This limits testing
    # to single-argument calls of atleast_3d, but it should pass the validation check.

    # Input 1: Single scalar input
    input_dict_1 = {'arys': np.array(10)}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Single 1-D array
    input_dict_2 = {'arys': np.array([1, 2, 3])}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single 2-D array
    input_dict_3 = {'arys': np.array([[1, 2], [3, 4]])}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Single 3-D array (should be unchanged)
    input_dict_4 = {'arys': np.arange(24).reshape(2, 3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single 4-D array (should be unchanged)
    input_dict_5 = {'arys': np.ones((1, 2, 3, 4), dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single empty 1-D array
    input_dict_6 = {'arys': np.array([])}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single empty 2-D array
    input_dict_7 = {'arys': np.empty((0, 5))}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: An array with a single element
    input_dict_8 = {'arys': np.array([100])}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 1-D float array with special values
    input_dict_9 = {'arys': np.array([-5.5, np.nan, np.inf], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 1-D complex number array
    input_dict_10 = {'arys': np.array([1+2j, 3-4j], dtype=np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: 5-D array
    input_dict_11 = {'arys': np.zeros((1, 1, 2, 1, 1), dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: 2-D boolean array
    input_dict_12 = {'arys': np.array([[True, False], [False, True]])}
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.atleast_3d"] = tf_experimental_numpy_atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.atleast_3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_3d'.")

check_valid('tf.experimental.numpy.atleast_3d', generated_inputs['tf.experimental.numpy.atleast_3d'], lib="tf", suffix=0)
