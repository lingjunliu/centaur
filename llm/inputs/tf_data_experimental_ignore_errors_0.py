
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tfd_experimental_ignore_errors_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.ignore_errors.
    The API returns a function to be used with `dataset.apply()`.
    The test harness requires a special 'inner_values' key to construct the
    base dataset for this function-returning API.
    The value for 'inner_values' is a tuple containing the numpy array(s),
    which is a valid format for tf.data.Dataset.from_tensor_slices.
    """
    list_of_inputs = []

    # Input 1: log_warning=False, inner_values as tuple
    input_dict = {
        'log_warning': False,
        'inner_values': (np.array([1., 2., 0., 4.], dtype=np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: log_warning=True, inner_values as tuple
    input_dict = {
        'log_warning': True,
        'inner_values': (np.array([5., 0., 6., 7.], dtype=np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D data
    input_dict = {
        'log_warning': False,
        'inner_values': (np.array([[1, 2], [3, 0], [5, 6]], dtype=np.int32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No errors
    input_dict = {
        'log_warning': True,
        'inner_values': (np.array([1, 2, 3], dtype=np.int64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All errors
    input_dict = {
        'log_warning': False,
        'inner_values': (np.array([0, 0, 0], dtype=np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64
    input_dict = {
        'log_warning': True,
        'inner_values': (np.array([-1.0, 0.0, 1.0], dtype=np.float64),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Empty input
    input_dict = {
        'log_warning': False,
        'inner_values': (np.array([], dtype=np.float32),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Another 2D with all errors
    input_dict = {
        'log_warning': True,
        'inner_values': (np.array([[0, 0], [0, 0]], dtype=np.int16),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Unsigned int
    input_dict = {
        'log_warning': False,
        'inner_values': (np.array([10, 20, 0, 30], dtype=np.uint8),)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D data
    inner_values_3d = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    inner_values_3d[1, 0, 1] = 0.
    input_dict = {
        'log_warning': True,
        'inner_values': (inner_values_3d,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.ignore_errors"] = tfd_experimental_ignore_errors_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.ignore_errors' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.ignore_errors'.")

check_valid('tf.data.experimental.ignore_errors', generated_inputs['tf.data.experimental.ignore_errors'], lib="tf", suffix=0)
