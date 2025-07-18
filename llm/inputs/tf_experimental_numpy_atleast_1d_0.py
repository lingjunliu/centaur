
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_atleast_1d_inputs():
    """
    Generates a list of valid inputs for tf.experimental.numpy.atleast_1d.
    The 'arys' parameter corresponds to *args, and the testing framework expects
    a single NumPy array that can be unpacked along its first dimension to
    represent the multiple arguments. This means all individual array arguments
    for a single test case must have the same shape.
    """
    list_of_inputs = []

    # Input 1: Single scalar input.
    # Passed as np.array([5]), unpacked to atleast_1d(5).
    input_dict_1 = {'arys': np.array([5])}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Two scalar inputs.
    # Passed as np.array([-10, 20]), unpacked to atleast_1d(-10, 20).
    input_dict_2 = {'arys': np.array([-10, 20])}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single 1-D array input.
    # Passed as np.array([[1, 2, 3]]), unpacked to atleast_1d([1, 2, 3]).
    input_dict_3 = {'arys': np.array([[1, 2, 3]])}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Two 1-D array inputs of the same shape.
    # Passed as a 2D array, unpacked to atleast_1d([1.1, 2.2], [3.3, 4.4]).
    input_dict_4 = {'arys': np.array([[1.1, 2.2], [3.3, 4.4]])}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single 2-D array input.
    # Passed as a 3D array, unpacked to atleast_1d([[1, 2], [3, 4]]).
    input_dict_5 = {'arys': np.array([[[1, 2], [3, 4]]])}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Three 2-D array inputs of the same shape.
    # Passed as a 3D array, unpacked to atleast_1d([[1,2]], [[3,4]], [[5,6]]).
    input_dict_6 = {'arys': np.array([[[1, 2]], [[3, 4]], [[5, 6]]])}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single empty 1-D array input.
    # Passed as np.array([[]]), unpacked to atleast_1d([]).
    input_dict_7 = {'arys': np.array([[]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Two empty 1-D array inputs.
    # Passed as np.array([[], []]), unpacked to atleast_1d([], []).
    input_dict_8 = {'arys': np.array([[], []], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Two complex number scalar inputs.
    # Passed as np.array([1+2j, 3-4j]), unpacked to atleast_1d(1+2j, 3-4j).
    input_dict_9 = {'arys': np.array([1 + 2j, 3 - 4j])}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Two 1-D boolean array inputs of the same shape.
    # Unpacked to atleast_1d([True, False], [False, True]).
    input_dict_10 = {'arys': np.array([[True, False], [False, True]])}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Single 3-D array input.
    input_dict_11 = {'arys': np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])}
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Three 1-D arrays with negative float values.
    input_dict_12 = {'arys': np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]])}
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.atleast_1d"] = tf_experimental_numpy_atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.atleast_1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.atleast_1d'.")

check_valid('tf.experimental.numpy.atleast_1d', generated_inputs['tf.experimental.numpy.atleast_1d'], lib="tf", suffix=0)
