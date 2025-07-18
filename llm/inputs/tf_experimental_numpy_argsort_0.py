
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_argsort_inputs():
    list_of_inputs = []

    # The error "ValueError: The `order` argument is not supported. Pass order=None"
    # conflicts with the required signature {'order': 'list'}.
    # The previous attempt to use `order=None` resulted in a `TypeError`,
    # indicating the testing environment cannot handle None for a 'list' type.
    # To satisfy the signature, `order` must be a list. The only possibility is an
    # empty list, `[]`. This will likely reproduce the ValueError, but it is the
    # only input that respects the provided type signature.

    # Input 1: Basic 1D array
    input_dict_1 = {
        'a': np.array([3, 1, 2, 5, 4]),
        'axis': -1,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D array with negative and duplicate values
    input_dict_2 = {
        'a': np.array([8, -2, 0, 5, -2, 10, 0]),
        'axis': 0,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D array with default axis (-1)
    input_dict_3 = {
        'a': np.array([[3, 1, 2], [6, 5, 4]]),
        'axis': -1,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D array with axis=0
    input_dict_4 = {
        'a': np.array([[3.0, 6.0], [1.0, 5.0], [2.0, 4.0]], dtype=np.float32),
        'axis': 0,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D array with axis=1
    input_dict_5 = {
        'a': np.random.rand(2, 3, 4).astype(np.float64),
        'axis': 1,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 1D array with floating point numbers
    input_dict_6 = {
        'a': np.array([1.1, -0.5, 3.14, 2.71, -0.5]),
        'axis': -1,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single element array
    input_dict_7 = {
        'a': np.array([100]),
        'axis': -1,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2D array of floats with negative axis
    input_dict_8 = {
        'a': np.array([[10.1, 9.9, 12.5], [1.1, 1.2, 1.0]]),
        'axis': -2,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large 1D array
    input_dict_9 = {
        'a': np.arange(100, 0, -1),
        'axis': 0,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D array
    input_dict_10 = {
        'a': np.random.uniform(size=(2,2,2,2)),
        'axis': 3,
        'kind': 'quicksort',
        'order': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.argsort"] = tf_experimental_numpy_argsort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.argsort'.")

check_valid('tf.experimental.numpy.argsort', generated_inputs['tf.experimental.numpy.argsort'], lib="tf", suffix=0)
